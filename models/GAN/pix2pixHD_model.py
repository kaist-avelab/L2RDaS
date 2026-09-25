import numpy as np
import torch
import os
from torch.autograd import Variable
from utils.l2r.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from utils.spconv_utils import replace_feature, spconv
import torch.nn.functional as F
from utils.bbox_trans.transform_util import points_in_rbbox

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

import open3d as o3d

def get_points_from_line(list_infos, is_with_arrow=False, length_arrow=1.0, length_tips=0.4):
    line_width=0.1
    x, y, z, azi_deg, l_2, w_2, h_2 = list_infos
    theta = azi_deg
    l = l_2
    w = w_2
    h = h_2

    points = [
        [l/2, w/2, h/2],            # 0
        [l/2, w/2, -h/2],           # 1
        [l/2, -w/2, h/2],           # 2
        [l/2, -w/2, -h/2],          # 3
        [-l/2, w/2, h/2],           # 4
        [-l/2, w/2, -h/2],          # 5
        [-l/2, -w/2, h/2],          # 6
        [-l/2, -w/2, -h/2],         # 7
    ]

    if is_with_arrow:
        points.extend([
            [0, 0, 0],                  # 8
            [l/2+length_arrow, 0, 0],   # 9
            [l/2+length_arrow-length_tips, length_tips, 0],     # 10
            [l/2+length_arrow-length_tips, -length_tips, 0],    # 11
        ])

    ### Rotation ###
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    mat_rot = np.array([
        [cos_th, -sin_th, 0],
        [sin_th, cos_th, 0],
        [0, 0, 1]
    ])
    points = list(map(lambda point: mat_rot.dot(np.array(point).reshape((3,1))).reshape(1,3).tolist()[0],points))

    ### Translation ###
    points = list(map(lambda point: [point[0]+x, point[1]+y, point[2]+z], points))


    lines = [
        [0, 1], [0, 2], [0, 4], [1, 3], \
        [1, 5], [2, 3], [2, 6], [3, 7], \
        [4, 5], [4, 6], [5, 7], [6, 7], \
    ]

    if is_with_arrow:
        lines.extend([
            [8, 9], [9, 10], [9, 11], \
        ])

    # Increase line thickness by adding points along the lines
    thick_line_points = []
    for start, end in lines:
        start_point = np.array(points[start])
        end_point = np.array(points[end])
        line_vector = end_point - start_point
        num_points = int(np.linalg.norm(line_vector) / line_width)
        for i in range(num_points + 1):
            thick_line_points.append(start_point + i * line_vector / num_points)

    return thick_line_points

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, l1_loss):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake, l1_loss),flags) if f]
        return loss_filter

    def initialize(self, opt, cfg):
        #----------------init for kradar preprocess---------------
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.dataset_cfg = cfg.DATASET

        # class
        self.num_class = 0
        self.class_names = []
        dict_label = self.cfg.DATASET.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key)
        self.dict_cls_name_to_id = dict()
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            self.dict_cls_name_to_id[k] = logit_idx
            self.dict_cls_name_to_id['Background'] = 0
            if logit_idx > 0:
                self.num_class += 1
                self.class_names.append(k)
        # print(self.class_names)

        # Common params
        num_point_features = self.dataset_cfg.ldr64.n_used
        self.num_point_features = num_point_features
        point_cloud_range = np.array(self.dataset_cfg.roi.xyz)
        voxel_size = self.dataset_cfg.roi.voxel_size
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        model_info_dict = dict(
            module_list = [],
            num_rawpoint_features = num_point_features,
            num_point_features = num_point_features,
            grid_size = grid_size,
            point_cloud_range = point_cloud_range,
            voxel_size = voxel_size,
        )
        self.voxel_generator_train = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features,
            max_num_points_per_voxel=self.dataset_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.dataset_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['train'],
        )
        self.voxel_generator_test = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            num_point_features=num_point_features,
            max_num_points_per_voxel=self.dataset_cfg.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            max_num_voxels=self.dataset_cfg.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['test'],
        )


        #----------------init GAN--------------
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            #netD_input_nc = input_nc + opt.output_nc
            netD_input_nc = 2 #TBD
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            self.criterionl1 = torch.nn.SmoothL1Loss()
            # self.criterionsmoothl1 = torch.nn.SmoothL1Loss()

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake', 'l1Loss')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False, dict_item = None):
        if self.opt.label_nc == 0:
            input_label=label_map

            # # # explictly RCS objects
            # bboxes = dict_item['meta'][0]['label'] #TODO: because batch 1
            # sedan_list =[]
            # bus_list=[]

            # for bbox in bboxes:
            #     class_name, [x, y, z, theta, xl, yl, zl], _, _ = bbox
            #     if class_name == "Sedan":
            #         sedan_list.append([x, y, z, theta, xl, yl, zl])
            #     elif class_name == "Bus or Truck" :
            #         bus_list.append([x, y, z, theta, xl, yl, zl])
            #     else:
            #         print("Add new class! in 'encode_input'")
            #         exit()

            # # Convert to numpy arrays
            # sedan_array = np.array(sedan_list)
            # bus_array = np.array(bus_list)
            # label_map_np = label_map.cpu().numpy()

            # # Initialize additional channels
            # sedan_mask_expanded = np.zeros((label_map_np.shape[0], 1), dtype=np.float32)
            # bus_mask_expanded = np.zeros((label_map_np.shape[0], 1), dtype=np.float32)
            # boundary_expanded = np.zeros((label_map_np.shape[0], 1), dtype=np.float32)

            # # Filter points where the last channel is not zero
            # valid_points_mask = label_map_np[:, -1] == 0.

            # # print(label_map_np[ori_points_num-1])
            # if sedan_array.size > 0:
            #     masks_sedan = points_in_rbbox(label_map_np[:,:3], sedan_array)
            #     sedan_mask_expanded[masks_sedan.any(-1)] = 255.0
            #     sedan_mask_expanded[valid_points_mask] = 0.0
            # #-----explictly sedan
            # if bus_array.size > 0:
            #     masks_bus = points_in_rbbox(label_map_np[:,:3], bus_array)
            #     bus_mask_expanded[masks_bus.any(-1)] = 255.0
            #     bus_mask_expanded[valid_points_mask] = 0.0
            # #-----explictly boundary
            # if dict_item['meta'][0]['num_obj']>0:
            #     boundary_expanded[valid_points_mask] = 255.0

            # # #-------Open3D로 시각화
            # # if True:
            # #     print(" masked visualize")
            # #     # all included point cloud
            # #     point_cloud_ = o3d.geometry.PointCloud()
            # #     # tmp = label_map_np[masks_sedan.any(-1)]
            # #     # valid_points_mask = tmp[:, -1] != 0.
            # #     # tmp = tmp[valid_points_mask]
            # #     tmp = label_map_np[valid_points_mask]
            # #     point_cloud_.points = o3d.utility.Vector3dVector(label_map_np[:,:3])
            # #     geometries = [point_cloud_]

            # #     # Create a visualizer object
            # #     vis = o3d.visualization.Visualizer()
            # #     vis.create_window()
            # #     for geometry in geometries:
            # #         vis.add_geometry(geometry)

            # #     vis.poll_events()
            # #     vis.update_renderer()

            # #         # Pause to visualize
            # #     vis.run()

            # #         # Close the visualizer window
            # #     vis.destroy_window()
            # #     #----------------------------



            # # Concatenate the masks with the original label_map
            # # input_label=label_map_np
            # input_label = np.concatenate((label_map_np, sedan_mask_expanded, bus_mask_expanded, boundary_expanded), axis=1)
            # #input_label = np.concatenate((label_map_np, sedan_mask_expanded, bus_mask_expanded), axis=1)
            # input_label = torch.from_numpy(input_label).cuda()

        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)

        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        # with torch.no_grad():
        #     input_label = Variable(input_label)

        #     # real images for training
        #     if real_image is not None:
        #         real_image = Variable(real_image.data.cuda())

        #     # instance map for feature encoding
        #     if self.use_features:
        #         # get precomputed feature maps
        #         if self.opt.load_features:
        #             feat_map = Variable(feat_map.data.cuda())
        #         if self.opt.label_feat:
        #             inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

    def encode_input_inference(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False, dict_item = None):
        if self.opt.label_nc == 0:
            input_label=label_map
            # # explictly RCS objects
            # bboxes = dict_item['meta']['label'] #TODO: because batch 1
            # sedan_list =[]
            # bus_list=[]

            # for bbox in bboxes:
            #     class_name, [x, y, z, theta, xl, yl, zl], _, _ = bbox
            #     if class_name == "Sedan":
            #         sedan_list.append([x, y, z, theta, xl, yl, zl])
            #     elif class_name == "Bus or Truck" :
            #         bus_list.append([x, y, z, theta, xl, yl, zl])
            #     else:
            #         print("Add new class! in 'encode_input'")
            #         exit()

            # # Convert to numpy arrays
            # sedan_array = np.array(sedan_list)
            # bus_array = np.array(bus_list)
            # # label_map_np = label_map.cpu().numpy()
            # label_map_np = label_map.numpy()

            #  # Initialize additional channels
            # sedan_mask_expanded = np.zeros((label_map_np.shape[0], 1), dtype=np.float32)
            # bus_mask_expanded = np.zeros((label_map_np.shape[0], 1), dtype=np.float32)
            # boundary_expanded = np.zeros((label_map_np.shape[0], 1), dtype=np.float32)  #bbox code #1

            # # Filter points where the last channel is not zero
            # valid_points_mask = label_map_np[:, -1] == 0.       #bbox code #2

            # if sedan_array.size > 0:
            #     masks_sedan = points_in_rbbox(label_map_np[:,:3], sedan_array)
            #     sedan_mask_expanded[masks_sedan.any(-1)] = 255.0
            #     sedan_mask_expanded[valid_points_mask] = 0.0  #bbox code #3
            # #-----explictly sedan
            # if bus_array.size > 0:
            #     masks_bus = points_in_rbbox(label_map_np[:,:3], bus_array)
            #     bus_mask_expanded[masks_bus.any(-1)] = 255.0
            #     bus_mask_expanded[valid_points_mask] = 0.0 #bbox code #4
            # #-----explictly boundary
            # if dict_item['meta']['num_obj']>0: #bbox code #5
            #     boundary_expanded[valid_points_mask] = 255.0 #bbox code #6

            # # #Concatenate the masks with the original label_map
            # # input_label=label_map_np
            # # input_label = np.concatenate((label_map_np, sedan_mask_expanded, bus_mask_expanded), axis=1)
            # input_label = np.concatenate((label_map_np, sedan_mask_expanded, bus_mask_expanded, boundary_expanded), axis=1) #bbox code #7
            # input_label = torch.from_numpy(input_label).cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)

        if infer:
            with torch.no_grad():
                input_label = Variable(input_label)
        else:
            input_label = Variable(input_label)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        # with torch.no_grad():
        #     input_label = Variable(input_label)

        #     # real images for training
        #     if real_image is not None:
        #         real_image = Variable(real_image.data.cuda())

        #     # instance map for feature encoding
        #     if self.use_features:
        #         # get precomputed feature maps
        #         if self.opt.load_features:
        #             feat_map = Variable(feat_map.data.cuda())
        #         if self.opt.label_feat:
        #             inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image,dict_item, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat) #[1,2,32,192,192]
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, infer=False, dict_datum=None, mask=None):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat, dict_item = dict_datum)

        #input label --------------> make tensor with sparse conv.
        voxel_features = input_label
        voxel_coords = dict_datum['voxel_coords']
        batch_size = dict_datum['batch_size']

        #(24, 192, 192) is shape ==> TODO
        spatial_shape=[33,192,192]

        sparse_input = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=spatial_shape, batch_size=batch_size)
        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)
            input_label=sparse_input.dense()
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = sparse_input
        fake_image = self.netG.forward(input_concat, dict_datum)

        if mask is not None:
            # fake_image = fake_image * mask
            # mask = mask.repeat(batch_size, 1, 1, 1, 1)
            # fake_image = fake_image * mask
            fake_image = torch.where(mask.bool(), fake_image, torch.tensor(0).to(fake_image.device))


        input_label=sparse_input.dense()
        input_label = input_label[:, 3, :-1, :, :].unsqueeze(1) #decrese z axis

        real_image=real_image.unsqueeze(0).unsqueeze(0)
        #<---------------------------

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, dict_datum, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False, mask)

        # Real Detection and Loss
        pred_real = self.discriminate(input_label, real_image, dict_datum)
        loss_D_real = self.criterionGAN(pred_real, True, mask)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True, mask)

        # Pixel-wise L1 loss
        loss_pixelwise_L1 = self.criterionl1(fake_image[mask==1], real_image[mask==1])

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
            #         # Feature map 크기에 맞게 마스크를 조정합니다.
            #         mask_resized = F.interpolate(mask, size=pred_fake[i][j].shape[2:], mode='nearest')
            #         # 손실을 계산하고 마스크를 곱합니다.
            #         l1_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
            #         masked_loss = l1_loss * mask_resized

            # # 마스크된 손실을 평균하여 최종 손실에 추가합니다.
            # loss_G_GAN_Feat += D_weights * feat_weights * masked_loss.mean() * self.opt.lambda_feat

                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image, mask) * self.opt.lambda_feat

        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_pixelwise_L1 ), None if not infer else fake_image ]

    def inference(self, label, inst, image=None, dict_item=None):
        # Encode Inputs
        if image is not None:
            image = image.cuda()
        # input_label = label.cuda()
        input_label = label
        inst = inst.cuda()
        input_label, inst_map, real_image, _ = self.encode_input_inference(input_label, inst, image, infer=True, dict_item=dict_item)

        #input label --------------> make tensor with sparse conv.
        voxel_features = input_label
        voxel_coords = torch.tensor(dict_item['voxel_coords']).int().cuda()

        batch_size = 1

        #(24, 192, 192) is shape ==> TODO
        spatial_shape=[33,192,192]

        batch_indices = torch.zeros((voxel_coords.shape[0], 1), dtype=torch.int32).cuda()
        voxel_coords = torch.cat([batch_indices, voxel_coords], dim=1)

        sparse_input = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords, spatial_shape=spatial_shape, batch_size=batch_size)

        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features
                feat_map = self.sample_features(inst_map)
            input_label=sparse_input.dense()
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = sparse_input

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat, dict_item)


        # #---vis---
        # import util.util as util
        # from collections import OrderedDict
        # from utils.l2r.visualizer import Visualizer
        # vis=Visualizer(self.opt)
        # input_label=sparse_input.dense()
        # input_label = input_label[:, 3, :, :, :].unsqueeze(1)
        # real_image = torch.tensor(dict_item['rdr_cube']).float().unsqueeze(0).unsqueeze(0)
        # visuals = OrderedDict([('input_label', util.tensor2label(input_label[0], self.opt.label_nc)),
        #                                     ('real_image', util.tensor2im(real_image[0]))])

        # vis.save_images(visuals, path)
        # #----
        return fake_image

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def pre_processor(self, batch_dict):
        if self.is_pre_processing is None:
            return batch_dict
        elif self.is_pre_processing == 'v1_0':
            # Shuffle (DataProcessor.shuffle_points)
            batched_ldr64 = batch_dict['ldr64']
            batched_indices_ldr64 = batch_dict['batch_indices_ldr64']
            list_points = []
            list_voxels = []
            list_voxel_coords = []
            list_voxel_num_points = []
            for batch_idx in range(batch_dict['batch_size']):
                temp_points = batched_ldr64[torch.where(batched_indices_ldr64 == batch_idx)[0],:self.num_point_features]

                if (self.shuffle_points) and (self.training):
                    shuffle_idx = np.random.permutation(temp_points.shape[0])
                    temp_points = temp_points[shuffle_idx,:]
                list_points.append(temp_points)

                if self.transform_points_to_voxels:
                    if self.training:
                        voxels, coordinates, num_points = self.voxel_generator_train.generate(temp_points.numpy())
                    else:
                        voxels, coordinates, num_points = self.voxel_generator_train.generate(temp_points.numpy())
                    voxel_batch_idx = np.full((coordinates.shape[0], 1), batch_idx, dtype=np.int64)
                    coordinates = np.concatenate((voxel_batch_idx, coordinates), axis=-1) # bzyx

                    list_voxels.append(voxels)
                    list_voxel_coords.append(coordinates)
                    list_voxel_num_points.append(num_points)

            batched_points = torch.cat(list_points, dim=0)
            batch_dict['points'] = torch.cat((batched_indices_ldr64.reshape(-1,1), batched_points), dim=1).cuda() # b, x, y, z, intensity
            batch_dict['voxels'] = torch.from_numpy(np.concatenate(list_voxels, axis=0)).cuda()
            batch_dict['voxel_coords'] = torch.from_numpy(np.concatenate(list_voxel_coords, axis=0)).cuda()
            batch_dict['voxel_num_points'] = torch.from_numpy(np.concatenate(list_voxel_num_points, axis=0)).cuda()
            batch_dict['gt_boxes'] = batch_dict['gt_boxes'].cuda()

            return batch_dict

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp, dict_item):
        label, inst = inp
        return self.inference(label, inst, dict_item=dict_item)
