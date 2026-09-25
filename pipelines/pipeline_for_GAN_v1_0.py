'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import torch
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import shutil
from torch.utils.data import Subset

# Ingnore numba warning
from numba.core.errors import NumbaWarning
import warnings
import logging
warnings.simplefilter('ignore', category=NumbaWarning)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)

from torch.utils.tensorboard import SummaryWriter

from utils.util_pipeline import *
from utils.util_point_cloud import *
from utils.util_config import cfg, cfg_from_yaml_file

from utils.util_point_cloud import Object3D
import utils.kitti_eval.kitti_common as kitti
from utils.kitti_eval.eval import get_official_eval_result

from utils.util_optim import clip_grad_norm_

#
from models.GAN.models import create_model
from utils.l2r.visualizer import Visualizer
import time
from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
import gc
from utils.spconv_utils import replace_feature, spconv
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
from scipy import linalg
import torch.nn.functional as F
import torch.nn as nn

#----vis code---
def visualize_point_cloud(points, title):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.visualization.draw_geometries([pcd], window_name=title)

def visualize_voxel_features(voxel_features, voxel_coords, spatial_shape, title):
    voxel_features = voxel_features.cpu().numpy()
    voxel_coords = voxel_coords.cpu().numpy()

    # Create a list to hold the Voxel centers and their corresponding colors
    voxel_centers = []
    voxel_colors = []

    # Assuming intensity is the fourth channel (index 3)
    intensity_idx = 3


    for i, coord in enumerate(voxel_coords):
        z, y, x = coord[1], coord[2], coord[3]
        center = [(x + 0.5) * spatial_shape[2] / 192.0,
                  (y + 0.5) * spatial_shape[1] / 192.0,
                  (z + 0.5) * spatial_shape[0] / 32.0]
        voxel_centers.append(center)

        # Normalize the intensity to [0, 1] for color mapping
        intensity = voxel_features[i, intensity_idx]
        intensity_normalized = (intensity - voxel_features[:, intensity_idx].min()) / (voxel_features[:, intensity_idx].max() - voxel_features[:, intensity_idx].min())
        # Map the normalized intensity to a color using plt.cm.jet
        color = plt.cm.jet(intensity_normalized)[:3]  # Extract RGB values
        voxel_colors.append(color)


    # Convert to numpy arrays
    voxel_centers = np.array(voxel_centers)
    voxel_colors = np.array(voxel_colors)

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    pcd.colors = o3d.utility.Vector3dVector(voxel_colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name=title)

def compare_point_cloud_and_voxel_features(batch_dict, dict_net):
    # Visualize original point cloud
    points = dict_net['ldr64']
    visualize_point_cloud(points, title="Original Point Cloud")

    # Visualize VFE voxel features
    voxel_features = dict_net['voxel_features']
    voxel_coords = dict_net['voxel_coords']
    spatial_shape = [32, 192, 192]  # Example spatial shape
    visualize_voxel_features(voxel_features, voxel_coords, spatial_shape, title="VFE Voxel Features")

#---

#---

class PipelineGAN_v1_0():
    def __init__(self, path_cfg=None, mode='train', option=False):
        '''
        * mode in ['train', 'test', 'vis']
        *   'train' denotes both train & test
        *   'test'  denotes mode for inference
        '''
        self.cfg = cfg_from_yaml_file(path_cfg, cfg)
        self.mode = mode
        self.update_cfg_regarding_mode()

        if self.cfg.GENERAL.SEED is not None:
            try:
                set_random_seed(cfg.GENERAL.SEED, cfg.GENERAL.IS_CUDA_SEED, cfg.GENERAL.IS_DETERMINISTIC)
            except:
                print('* Exception error: check cfg.GENERAL for seed')
                set_random_seed(cfg.GENERAL.SEED)


        # -----------l2r pipeline---------

        self.opt=option
        self.start_epoch=1
        self.epoch_iter=0
        self.l2r_generate = self.cfg.DATASET.object_sample.make_l2r_tensor
        #-------------
        print('* K-Radar dataset is being loaded.')
        self.dataset_train = build_dataset(self, split='train', option=self.opt) if self.mode == 'train' else None
        self.dataset_test = build_dataset(self, split='test', option=self.opt)
        print('* The dataset is loaded.')
        if mode == 'train': # for setting scheduler
            self.cfg.DATASET.NUM = len(self.dataset_train)
        elif mode in ['test', 'vis']:
            self.cfg.DATASET.NUM = len(self.dataset_test)
        # print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) # check if it is updated
        self.iter_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'iter.txt')

        # self.network = build_network(self).cuda()
        # self.optimizer = build_optimizer(self, self.network)
        # self.scheduler = build_scheduler(self, self.optimizer)
        self.epoch_start = 0
        self.dataset_size=self.cfg.DATASET.NUM
        self._network = build_network(self).cuda()
        self.network = create_model(self.opt, self.cfg)
        self.visualizer = Visualizer(self.opt)
        if self.opt.fp16:
            from apex import amp
            self.network, [self.optimizer_G, self.optimizer_D] = amp.initialize(self.network, [self.network.optimizer_G, self.network.optimizer_D], opt_level='O1')
            self.network = torch.nn.DataParallel(self.network, device_ids=self.opt.gpu_ids)
        else:
            self.optimizer_G, self.optimizer_D = self.network.module.optimizer_G, self.network.module.optimizer_D


        self.total_steps = (self.start_epoch-1) * self.dataset_size + self.epoch_iter

        self.display_delta = self.total_steps % self.opt.display_freq
        self.print_delta = self.total_steps % self.opt.print_freq
        self.save_delta = self.total_steps % self.opt.save_latest_freq

        # Logging
        if self.cfg.GENERAL.LOGGING.IS_LOGGING:
            self.set_logging(path_cfg)

        # Validation
        if self.cfg.VAL.IS_VALIDATE:
            self.set_validate()
        else:
            self.is_validate = False

        if self.cfg.GENERAL.RESUME.IS_RESUME:
            self.resume_network()

        self.cfg_dataset_ver2 = self.cfg.get('cfg_dataset_ver2', False)
        self.get_loss_from = self.cfg.get('get_loss_from', 'head')
        self.optim_fastai = True \
            if self.cfg.OPTIMIZER.NAME in ['adam_onecycle', 'adam_cosineanneal'] else False
        self.grad_norm_clip = self.cfg.OPTIMIZER.get('GRAD_NORM_CLIP', -1)

        # Vis
        self.set_vis()

        # self.show_pline_description()
        #--for FID

    def update_cfg_regarding_mode(self):
        '''
        * You don't have to update values in cfg changed in dataset
        * They are related in pointer
        * e.g., check print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) after dataset initialization
        '''
        if self.mode == 'train':
            pass
        elif self.mode == 'test':
            self.cfg.OPTIMIZER.NUM_WORKERS = 0
        elif self.mode == 'vis':
            self.cfg.OPTIMIZER.NUM_WORKERS = 0
            self.cfg.GET_ITEM = {
                'rdr_sparse_cube'   : True,
                'rdr_tesseract'     : False,
                'rdr_cube'          : True,
                'rdr_cube_doppler'  : False,
                'ldr_pc_64'         : True,
                'cam_front_img'     : True,
            }
        else:
            print('* Exception error (Pipeline): check modify_cfg')
        return

    def set_validate(self):
        self.is_validate = True
        self.is_consider_subset = self.cfg.VAL.IS_CONSIDER_VAL_SUBSET
        self.val_per_epoch_subset = self.cfg.VAL.VAL_PER_EPOCH_SUBSET
        self.val_num_subset = self.cfg.VAL.NUM_SUBSET
        self.val_per_epoch_full = self.cfg.VAL.VAL_PER_EPOCH_FULL

        self.val_keyword = self.cfg.VAL.CLASS_VAL_KEYWORD # for kitti_eval
        list_val_keyword_keys = list(self.val_keyword.keys()) # same order as VAL.CLASS_VAL_KEYWORD.keys()
        self.list_val_care_idx = []

        # index matching with kitti_eval
        for cls_name in self.cfg.VAL.LIST_CARE_VAL:
            idx_val_cls = list_val_keyword_keys.index(cls_name)
            self.list_val_care_idx.append(idx_val_cls)
        # print(self.list_val_care_idx)

        ### Consider output of network and dataset ###
        if self.cfg.VAL.REGARDING == 'anchor':
            self.val_regarding = 0 # anchor
            self.list_val_conf_thr = self.cfg.VAL.LIST_VAL_CONF_THR
        else:
            print('* Exception error: check VAL.REGARDING')
        ### Consider output of network and dataset ###

    def set_vis(self):
        if self.cfg_dataset_ver2:
            pass # TODO
        else:
            self.dict_cls_name_to_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
            self.dict_cls_id_to_name = dict()
            for k, v in self.dict_cls_name_to_id.items():
                if v != -1:
                    self.dict_cls_id_to_name[v] = k
            self.dict_cls_name_to_bgr = self.cfg.VIS.CLASS_BGR
            self.dict_cls_name_to_rgb = self.cfg.VIS.CLASS_RGB

    def show_pline_description(self):
        print('* newtork (description start) -------')
        print(self.network)
        print('* newtork (description end) ---------')
        print('* optimizer (description start) -----')
        print(self.optimizer)
        print('* optimizer (description end) -------')
        print(f'* mode = {self.mode}')
        len_data = self.cfg.DATASET.NUM
        print(f'* dataset length = {len_data}')

    def set_logging(self, path_cfg, is_print_where=True):
        self.is_logging = True
        str_local_time = get_local_time_str()
        str_exp = 'exp_' + str_local_time + '_' + self.cfg.GENERAL.NAME
        self.path_log = os.path.join(self.cfg.GENERAL.LOGGING.PATH_LOGGING, str_exp)
        if is_print_where:
            print(f'* Start logging in {str_exp}')
        if not (os.path.exists(self.path_log)):
            os.makedirs(self.path_log)
        else:
            print('* Exception error (Pipeline): same folder exists, try again')
            exit()

        self.log_train_iter = SummaryWriter(os.path.join(self.path_log, 'train_iter'), comment='iteration')
        self.log_train_epoch = SummaryWriter(os.path.join(self.path_log, 'train_epoch'), comment='epoch')
        self.log_test = SummaryWriter(os.path.join(self.path_log, 'test'), comment='test')
        self.log_iter_start = None

        self.is_save_model = self.cfg.GENERAL.LOGGING.IS_SAVE_MODEL
        try:
            self.interval_epoch_model = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_MODEL
            self.interval_epoch_util = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_UTIL
        except:
            self.interval_epoch_model = 1
            self.interval_epoch_util = 5
            print('* Exception error (Pipeline): check LOGGING.INTERVAL_EPOCH_MODEL/UTIL')
        if self.is_save_model:
            os.makedirs(os.path.join(self.path_log, 'models'))
            os.makedirs(os.path.join(self.path_log, 'utils'))

        # cfg backup (same files, just for identification)
        name_file_origin = path_cfg.split('/')[-1] # original cfg file name
        name_file_cfg = 'config.yml'
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_origin))
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_cfg))

        # code backup (TBD)

    def resume_network(self):
        path_exp = self.cfg.GENERAL.RESUME.PATH_EXP
        path_state_dict = os.path.join(path_exp, 'utils')
        epoch = self.cfg.GENERAL.RESUME.START_EP
        list_epochs = sorted(list(map(lambda x: int(x.split('.')[0].split('_')[1]), os.listdir(path_state_dict))))
        epoch = list_epochs[-1] if epoch is None else epoch

        path_state_dict = os.path.join(path_state_dict, f'util_{epoch}.pt')
        print('* Start resume, path_state_dict =  ', path_state_dict)
        state_dict = torch.load(path_state_dict)

        try:
            self.epoch_start = epoch + 1
            self.network.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.log_iter_start = state_dict['idx_log_iter']
            print(f'* Network & Optimizer are loaded / Resume epoch is {epoch} / Start from {self.epoch_start} ...')
        except:
            print('* Exception error (Pipeline): check resume network')
            exit()

        if ('scheduler_state_dict' in state_dict.keys()) and (not (self.scheduler is None)):
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            print('* Scheduler is loaded')
        else:
            print('* Scheduler is started from vanilla')

        ### Copy logging folder ###
        list_copy_dirs = ['train_epoch', 'train_iter', 'test', 'test_kitti']
        if (self.cfg.GENERAL.RESUME.IS_COPY_LOGS) and (self.is_logging):
            for copy_dir in list_copy_dirs:
                shutil.copytree(os.path.join(path_exp, copy_dir), \
                    os.path.join(self.path_log, copy_dir), dirs_exist_ok=True)
        ### Copy logging folder ###

        return

    def create_mask(self, data):
        data=data.unsqueeze(0).unsqueeze(0).clone()
        self.mask = (data != 0).float()

    def calculate_psnr_ssim(self, original, compressed):
        if isinstance(original, list):
            image_numpy_ori = []
            image_numpy_com = []
            for i in range(len(original)):
                image_numpy_ori.append(self.calculate_psnr(original[i], compressed[i]))
            return image_numpy_ori
        image_numpy_ori = original.cpu().float().numpy()
        image_numpy_com = compressed.cpu().float().numpy()

        # Remove the height axis by averaging over the height dimension (BEV)
        image_numpy_ori_bev = np.mean(image_numpy_ori[0], axis=0)  # shape will be (192, 192)
        image_numpy_com_bev = np.mean(image_numpy_com[0], axis=0)  # shape will be (192, 192)


        # Calculate PSNR
        mse = np.mean((image_numpy_ori_bev - image_numpy_com_bev) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        # Calculate SSIM
        ssim_value, _ = ssim(image_numpy_ori_bev, image_numpy_com_bev, full=True)

        return psnr, ssim_value

    def convert_3d_to_2d(self, tensor_3d):
        """
        3D 텐서를 2D 텐서로 변환합니다.
        (Batch, Depth, Height, Width) -> (Batch * Depth, Height, Width)
        """
        B, D, H, W = tensor_3d.shape
        tensor_2d = tensor_3d.permute(1, 0, 2, 3).contiguous().view(B * D, 1, H, W)
        return tensor_2d.clone().detach()

    def calculate_activation_statistics(self, images, model, batch_size=32, dims=2048, cuda=False):
        """ 이미지에서 특징을 추출한 후, 평균과 공분산을 계산합니다. """
        act = self.get_activations(images, model, batch_size, dims, cuda)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def convert_to_rgb(self, tensor_2d):
        """
        1채널 2D 텐서를 3채널 2D 텐서로 변환합니다.
        """
        return tensor_2d.repeat(1, 3, 1, 1)

    def resize_images(self, images, size=(300, 300)):
        """
        이미지를 주어진 크기로 리사이즈합니다.
        """
        resized_images = F.interpolate(images, size=size, mode='bilinear', align_corners=False)
        return resized_images

    def get_activations(self, images, model, batch_size=32, dims=2048, cuda=False):
        """ Inception v3 모델을 사용하여 이미지에서 특징을 추출합니다. """
        model.eval()
        if cuda:
            model.cuda()

        # 이미지 리스트의 길이를 직접 계산합니다.
        total_images = sum([img.size(0) for img in images])
        pred_arr = np.empty((total_images, dims))

        start_idx = 0

        for img_list in images:
            dataloader = torch.utils.data.DataLoader(img_list, batch_size=batch_size, shuffle=False, drop_last=False)
            for batch in tqdm(dataloader):
                if cuda:
                    batch = batch.cuda()
                with torch.no_grad():
                    batch_rgb = self.convert_to_rgb(batch)
                    batch_rgb = self.resize_images(batch_rgb)
                    pred = model(batch_rgb).cpu().numpy()  # squeeze 제거한 부분
                pred_arr[start_idx:start_idx + pred.shape[0]] = pred
                start_idx = start_idx + pred.shape[0]

        return pred_arr

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ 두 다변량 정규 분포 사이의 Fréchet 거리를 계산합니다. """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print("fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


    # 3D 텐서에서 FID 계산
    def calculate_fid_3d(self, real_images, fake_images, batch_size=32, cuda=False, dims=2048):
        real_images_2d = [self.convert_3d_to_2d(images) for images in real_images]
        fake_images_2d = [self.convert_3d_to_2d(images) for images in fake_images]
        mu1, sigma1 = self.calculate_activation_statistics(real_images_2d, self.model, batch_size, dims, cuda)
        mu2, sigma2 = self.calculate_activation_statistics(fake_images_2d, self.model, batch_size, dims, cuda)
        fid_value = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value

    def train_network(self, is_shuffle=True):
        # self.network.train()
        data_loader_train = torch.utils.data.DataLoader(self.dataset_train, \
            batch_size = self.cfg.OPTIMIZER.BATCH_SIZE, shuffle = is_shuffle, \
            collate_fn = self.dataset_train.collate_fn,
            num_workers = self.cfg.OPTIMIZER.NUM_WORKERS, drop_last = False, pin_memory=True)

        epoch_start = self.epoch_start
        epoch_end = self.cfg.OPTIMIZER.MAX_EPOCH

        if self.is_logging:
            idx_log_iter = 0 if self.log_iter_start is None else self.log_iter_start

        if self.optim_fastai:
            accumulated_iter = 0
            cfg_optim = self.cfg.OPTIMIZER
            use_amp = cfg_optim.get('USE_AMP', False)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=cfg_optim.get('LOSS_SCALE_FP16', 2.0**16))

        for epoch in range(epoch_start, epoch_end):
            print(f'* Training epoch = {epoch}/{epoch_end-1}')
            if self.is_logging:
                print(f'* Logging path = {self.path_log}')

            self.epoch_start_time = time.time()
            if epoch != epoch_start:
                self.epoch_iter = self.epoch_iter % self.dataset_size

            # self.network.train()
            # self.network.training = True
            avg_loss = []
            seq_name_list =[]
            seq_idx_list=[]
            for i, data in enumerate(tqdm(data_loader_train)):
                #------------reference with pix2pixhd
                if self.total_steps % self.opt.print_freq == self.print_delta:
                    iter_start_time = time.time()
                self.total_steps += self.opt.batchSize
                self.epoch_iter += self.opt.batchSize

                # whether to collect output images
                save_fake = self.total_steps % self.opt.display_freq == self.display_delta
                # save_fake=True
                ############## Forward Pass ######################
                #caz batch size is only 1
                dict_net = self._network(data)
                label=dict_net['voxel_features']
                inst=torch.tensor([0]) #TBD TODO
                image = dict_net['rdr_cube']
                feat=torch.tensor([0]) #TBD TODO

                #--vis
                # compare_point_cloud_and_voxel_features(dict_net, dict_net)
                #--

                self.create_mask(image)

                losses, generated = self.network(Variable(label), Variable(inst),
                    Variable(image), Variable(feat), infer=save_fake, dict_datum=dict_net, mask=self.mask)

                # sum per device losses
                losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
                loss_dict = dict(zip(self.network.module.loss_names, losses))

                # calculate final loss scalar
                loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
                loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) + loss_dict.get('pixelwise_L1', 0)

                ############### Backward Pass ####################
                # update generator weights
                self.optimizer_G.zero_grad()
                if self.opt.fp16:
                    with amp.scale_loss(loss_G, self.optimizer_G) as scaled_loss: scaled_loss.backward()
                else:
                    # torch.cuda.empty_cache()  # 에포크 종료 후 캐시 비우기
                    loss_G.backward()
                self.optimizer_G.step()

                # update discriminator weights
                self.optimizer_D.zero_grad()
                if self.opt.fp16:
                    with amp.scale_loss(loss_D, self.optimizer_D) as scaled_loss: scaled_loss.backward()
                else:
                    # torch.cuda.empty_cache()  # 에포크 종료 후 캐시 비우기
                    loss_D.backward()
                self.optimizer_D.step()

                ############## Display results and errors ##########
                ### print out errors
                if self.total_steps % self.opt.print_freq == self.print_delta:
                    errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                    t = (time.time() - iter_start_time) / self.opt.print_freq
                    self.visualizer.print_current_errors(epoch, self.epoch_iter, errors, t)
                    self.visualizer.plot_current_errors(errors, self.total_steps)
                    #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

                ### display output images
                if save_fake:
                    #input label --------------> make tensor with sparse conv.
                    voxel_features = label
                    voxel_coords = dict_net['voxel_coords']
                    batch_size = dict_net['batch_size']

                    #(24, 192, 192) is shape ==> TODO
                    spatial_shape=[32,192,192]

                    sparse_input = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=spatial_shape, batch_size=batch_size)
                    input_label=sparse_input.dense()
                    input_label = input_label[:, 3, :, :, :].unsqueeze(1)

                    real_image=image.unsqueeze(0).unsqueeze(0)

                    visuals = OrderedDict([('input_label', util.tensor2label(input_label[0], self.opt.label_nc, roi=[0, -38.4, -2, 76.8, 38.4, 10.8], dict_item=dict_net)),
                                        ('synthesized_image', util.tensor2im(generated.data[0], roi=[0, -38.4, -2, 76.8, 38.4, 10.8], dict_item=dict_net)),
                                        ('real_image', util.tensor2im(real_image[0], roi=[0, -38.4, -2, 76.8, 38.4, 10.8], dict_item=dict_net))])
                    self.visualizer.display_current_results(visuals, epoch, self.total_steps)

                    psnr_value, ssim_value = self.calculate_psnr_ssim(generated.data[0], real_image[0])
                    print(f"PSNR 값: {psnr_value} dB")
                    print(f"SSIM 값: {ssim_value} dB")

                    #-----calculate FID
                    # fid_value = self.calculate_fid_3d(real_image,generated.data, batch_size=1, cuda=True )
                    # print('FID:', fid_value)
                    # if fid_value < 100:
                    #     seq_name = dict_net['meta'][0]['seq']
                    #     idx_name = dict_net['meta'][0]['idx']
                    #     seq_name_list.append(seq_name)
                    #     seq_idx_list.append(idx_name)
                    #     print("fid_value under 100!", seq_name, idx_name)

                ### save latest model
                if self.total_steps % self.opt.save_latest_freq == self.save_delta:
                    print('saving the latest model (epoch %d, total_steps %d)' % (epoch, self.total_steps))
                    self.network.module.save('latest')
                    np.savetxt(self.iter_path, (epoch, self.epoch_iter), delimiter=',', fmt='%d')

                if self.epoch_iter >= self.dataset_size:
                    break


                # # free memory (Killed error, checked with htop)
                # if 'pointer' in data.keys():
                #     for dict_item in data['pointer']:
                #         for k in dict_item.keys():
                #             if k != 'meta':
                #                 dict_item[k] = None
                # for temp_key in data.keys():
                #     data[temp_key] = None

                # gc.collect()

            print("seq: ", len(seq_name_list), seq_name_list)
            print("idx: ", len(seq_idx_list), seq_idx_list)


            # end of epoch
            iter_end_time = time.time()
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, self.opt.niter + self.opt.niter_decay, time.time() - self.epoch_start_time))

            ### save model for this epoch
            if epoch % self.opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, self.total_steps))
                self.network.module.save('latest')
                self.network.module.save(epoch)
                np.savetxt(self.iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

            ### instead of only training the local enhancer, train the entire network after certain iterations
            if (self.opt.niter_fix_global != 0) and (epoch == self.opt.niter_fix_global):
                self.network.module.update_fixed_params()

            ### linearly decay learning rate after certain iterations
            if epoch > self.opt.niter:
                self.network.module.update_learning_rate()

    def load_dict_model(self, path_dict_model, is_strict=False):
        pt_dict_model = torch.load(path_dict_model)
        self.network.load_state_dict(pt_dict_model, strict=is_strict)

    # V2
    def vis_infer(self, sample_indices, conf_thr=0.7, is_nms=True, vis_mode=['lpc', 'spcube', 'cube'], is_train=False):
        '''
        * sample_indices: e.g. [0, 1, 2, 3, 4]
        * assume batch_size = 1 for convenience
        * vis_mode (TBD)
        '''
        self.network.eval()

        if is_train:
            dataset_loaded = self.dataset_train
        else:
            dataset_loaded = self.dataset_test
        subset = Subset(dataset_loaded, sample_indices)
        data_loader = torch.utils.data.DataLoader(subset,
                batch_size = 1, shuffle = False,
                collate_fn = self.dataset_test.collate_fn,
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)

        for dict_datum in data_loader:
            dict_out = self.network(dict_datum)
            dict_out = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms)

            ### Vis data ###
            pc_lidar = dict_datum['ldr64']
            # rdr_spcube = dict_datum['rdr_sparse_cube']
            # rdr_cube = dict_datum['rdr_cube']
            ### Vis data ###

            ### Labels ###
            labels = dict_out['label'][0]
            list_obj_label = []
            for label_obj in labels:
                cls_name, cls_id, (xc, yc, zc, rot, xl, yl, zl), obj_idx = label_obj
                obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                list_obj_label.append(obj)
            ### Labels ###

            ### Preds: post processing bbox ###
            list_obj_pred = []
            list_cls_pred = []
            if dict_datum['pp_num_bbox'] == 0:
                pass
            else:
                pp_cls = dict_datum['pp_cls']
                for idx_pred, pred_obj in enumerate(dict_datum['pp_bbox']):
                    conf_score, xc, yc, zc, xl, yl, zl, rot = pred_obj
                    obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                    list_obj_pred.append(obj)
                    list_cls_pred.append('Sedan')
                    # list_cls_pred.append(self.dict_cls_id_to_name[pp_cls[idx_pred]])
            ### Preds: post processing bbox ###

            ### Vis for open3d ###
            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [6, 7], #[5, 6],[4, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    [0, 2], [1, 3], [4, 6], [5, 7]]
            colors_label = [[0, 0, 0] for _ in range(len(lines))]
            list_line_set_label = []
            list_line_set_pred = []
            for label_obj in list_obj_label:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(label_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors_label)
                list_line_set_label.append(line_set)

            for idx_pred, pred_obj in enumerate(list_obj_pred):
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(pred_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                # colors_pred = [self.dict_cls_name_to_rgb[list_cls_pred[idx_pred]] for _ in range(len(lines))]
                colors_pred = [[1.,0.,0.] for _ in range(len(lines))]
                line_set.colors = o3d.utility.Vector3dVector(colors_pred)
                list_line_set_pred.append(line_set)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
            o3d.visualization.draw_geometries([pcd] + list_line_set_label + list_line_set_pred)
            ### Vis for open3d ###

        return list_obj_label, list_obj_pred

    # V2
    def validate_kitti(self, epoch=None, list_conf_thr=None, is_subset=False):
        self.network.training=False
        self.network.eval()

        eval_ver2 = self.cfg.get('cfg_eval_ver2', False)
        if eval_ver2:
            class_names = []
            dict_label = self.dataset_test.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                if logit_idx > 0:
                    class_names.append(k)
            self.dict_cls_id_to_name = dict()
            for idx_cls, cls_name in enumerate(class_names):
                self.dict_cls_id_to_name[(idx_cls+1)] = cls_name # 1 for Background


        ### Check is_validate with small dataset ###
        if is_subset:
            is_shuffle = True
            tqdm_bar = tqdm(total=self.val_num_subset, desc='* Test (Subset): ')
            log_header = 'val_sub'
        else:
            is_shuffle = False
            tqdm_bar = tqdm(total=len(self.dataset_test), desc='* Test (Total): ')
            log_header = 'val_tot'

        data_loader = torch.utils.data.DataLoader(self.dataset_test, \
                batch_size=1, shuffle=is_shuffle, collate_fn=self.dataset_test.collate_fn, \
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)

        if epoch is None:
            dir_epoch = 'none'
        else:
            dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

        # initialize via VAL.LIST_VAL_CONF_THR
        path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
        # print(path_dir)
        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'val.txt', 'w') as f:
                f.write('')
            f.close()

        for idx_datum, dict_datum in enumerate(data_loader):
            if is_subset & (idx_datum >= self.val_num_subset):
                break

            try:
                dict_out = self.network(dict_datum) # inference
                is_feature_inferenced = True
            except:
                print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                print('* Meta info: ', dict_out['meta'])
                is_feature_inferenced = False

            idx_name = str(idx_datum).zfill(6)

            ### for every conf in list_conf_thr ###
            for conf_thr in list_conf_thr:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'val.txt'
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                if is_feature_inferenced:
                    if eval_ver2:
                        pred_dicts = dict_out['pred_dicts'][0]
                        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                        list_pp_bbox = []
                        list_pp_cls = []

                        for idx_pred in range(len(pred_labels)):
                            x, y, z, l, w, h, th = pred_boxes[idx_pred]
                            score = pred_scores[idx_pred]

                            if score > conf_thr:
                                cls_idx = int(np.round(pred_labels[idx_pred]))
                                cls_name = class_names[cls_idx-1]
                                list_pp_bbox.append([score, x, y, z, l, w, h, th])
                                list_pp_cls.append(cls_idx)
                            else:
                                continue
                        pp_num_bbox = len(list_pp_cls)
                        dict_out_current = dict_out
                        dict_out_current.update({
                            'pp_bbox': list_pp_bbox,
                            'pp_cls': list_pp_cls,
                            'pp_num_bbox': pp_num_bbox,
                            'pp_desc': dict_out['meta'][0]['desc']
                        })
                    else:
                        dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                else:
                    dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)
                if dict_out is None:
                    print('* Exception error (Pipeline): dict_item is None in validation')
                    continue

                dict_out = dict_datum_to_kitti(self, dict_out)

                if len(dict_out['kitti_gt']) == 0: # no eval for emptry obj label
                    pass
                else:
                    ### Gt ###
                    for idx_label, label in enumerate(dict_out['kitti_gt']):
                        open_mode = 'w' if idx_label == 0 else 'a'
                        with open(labels_dir + '/' + idx_name + '.txt', open_mode) as f:
                            f.write(label+'\n')
                    ### Gt ###

                    ### Process description ###
                    with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out['kitti_desc'])
                    ### Process description ###

                    ### Pred: do not care len 0 with if else: already care as dummy ###
                    for idx_pred, pred in enumerate(dict_out['kitti_pred']):
                        open_mode = 'w' if idx_pred == 0 else 'a'
                        with open(preds_dir + '/' + idx_name + '.txt', open_mode) as f:
                            f.write(pred+'\n')
                    ### Pred: do not care len 0 with if else: already care as dummy ###

                    str_log = idx_name + '\n'
                    with open(split_path, 'a') as f:
                        f.write(str_log)

            # free memory (Killed error, checked with htop)
            if 'pointer' in dict_datum.keys():
                for dict_item in dict_datum['pointer']:
                    for k in dict_item.keys():
                        if k != 'meta':
                            dict_item[k] = None
            for temp_key in dict_datum.keys():
                dict_datum[temp_key] = None
            tqdm_bar.update(1)
        tqdm_bar.close()

        ### Validate per conf ###
        for conf_thr in list_conf_thr:
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
            split_path = path_dir + f'/{conf_thr}/' + 'val.txt'

            dt_annos = kitti.get_label_annos(preds_dir)
            val_ids = read_imageset_file(split_path)
            gt_annos = kitti.get_label_annos(labels_dir, val_ids)

            list_metrics = []
            for idx_cls_val in self.list_val_care_idx:
                dict_metrics, result_log = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                print(f'-----conf{conf_thr}-----')
                print(result_log)
                list_metrics.append(dict_metrics)

            for dict_metrics in list_metrics:
                cls_name = dict_metrics['cls']
                ious = dict_metrics['iou']
                bevs = dict_metrics['bev']
                ap3ds = dict_metrics['3d']
                self.log_test.add_scalars(f'{log_header}/BEV_conf_thr_{conf_thr}', {
                    f'iou_{ious[0]}_{cls_name}': bevs[0],
                    f'iou_{ious[1]}_{cls_name}': bevs[1],
                    f'iou_{ious[2]}_{cls_name}': bevs[2],
                }, epoch)
                self.log_test.add_scalars(f'{log_header}/3D_conf_thr_{conf_thr}', {
                    f'iou_{ious[0]}_{cls_name}': ap3ds[0],
                    f'iou_{ious[1]}_{cls_name}': ap3ds[1],
                    f'iou_{ious[2]}_{cls_name}': ap3ds[2],
                }, epoch)
        ### Validate per conf ###

    def validate_kitti_conditional(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False):
        self.network.eval()

        eval_ver2 = self.cfg.get('cfg_eval_ver2', False)
        if eval_ver2:
            class_names = []
            dict_label = self.dataset_test.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                if logit_idx > 0:
                    class_names.append(k)
            self.dict_cls_id_to_name = dict()
            for idx_cls, cls_name in enumerate(class_names):
                self.dict_cls_id_to_name[(idx_cls+1)] = cls_name # 1 for Background

        road_cond_list = ['urban', 'highway', 'countryside', 'alleyway', 'parkinglots', 'shoulder', 'mountain', 'university']
        time_cond_list = ['day', 'night']
        weather_cond_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']

        # Check is_validate with small dataset
        if is_subset:
            is_shuffle = True
            tqdm_bar = tqdm(total=self.val_num_subset, desc='Test (Subset): ')
        else:
            is_shuffle = False
            tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')

        data_loader = torch.utils.data.DataLoader(self.dataset_test, \
                batch_size = 1, shuffle = is_shuffle, collate_fn = self.dataset_test.collate_fn, \
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)

        if epoch is None:
            dir_epoch = 'none'
        else:
            dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

        # initialize via VAL.LIST_VAL_CONF_THR
        path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)

            os.makedirs(os.path.join(path_dir, f'{conf_thr}', 'all'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'all/val.txt', 'w') as f:
                f.write('')

            for road_cond in road_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', road_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + road_cond + '/val.txt', 'w') as f:
                    f.write('')

            for time_cond in time_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', time_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + time_cond + '/val.txt', 'w') as f:
                    f.write('')

            for weather_cond in weather_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', weather_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + weather_cond + '/val.txt', 'w') as f:
                    f.write('')

            pred_dir_list = []
            label_dir_list = []
            desc_dir_list = []
            split_path_list = []

            ### For All Conditions ###
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
            list_dir = [preds_dir, labels_dir, desc_dir]
            split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

            for temp_dir in list_dir:
                os.makedirs(temp_dir, exist_ok=True)

            pred_dir_list.append(preds_dir)
            label_dir_list.append(labels_dir)
            desc_dir_list.append(desc_dir)
            split_path_list.append(split_path)

            ### For Specific Conditions ###
            for road_cond in road_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + road_cond +'/val.txt'

                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)

            for time_cond in time_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + time_cond +'/val.txt'

                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)

            for weather_cond in weather_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + weather_cond +'/val.txt'

                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)

        # Creating gts and preds txt files for evaluation
        for idx_datum, dict_datum in enumerate(data_loader):
            if is_subset & (idx_datum >= self.val_num_subset):
                break

            try:
                dict_out = self.network(dict_datum) # inference
                is_feature_inferenced = True
            except:
                print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                print('* Meta info: ', dict_out['meta'])
                is_feature_inferenced = False

            if is_print_memory:
                print('max_memory: ', torch.cuda.max_memory_allocated(device='cuda'))

            idx_name = str(idx_datum).zfill(6)

            road_cond_tag, time_cond_tag, weather_cond_tag = \
                dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
            # print(dict_out['desc'][0])

            ### for every conf in list_conf_thr ###
            for conf_thr in list_conf_thr:
                ### For All Conditions ###
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                preds_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'preds')
                labels_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'gts')
                desc_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'desc')
                split_path_road =path_dir + f'/{conf_thr}/' + road_cond_tag + '/val.txt'

                preds_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'preds')
                labels_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'gts')
                desc_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'desc')
                split_path_time = path_dir + f'/{conf_thr}/' + time_cond_tag + '/val.txt'

                preds_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'preds')
                labels_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'gts')
                desc_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'desc')
                split_path_weather =path_dir + f'/{conf_thr}/' + weather_cond_tag + '/val.txt'

                os.makedirs(labels_dir_road, exist_ok=True)
                os.makedirs(labels_dir_time, exist_ok=True)
                os.makedirs(labels_dir_weather, exist_ok=True)
                os.makedirs(desc_dir_road, exist_ok=True)
                os.makedirs(desc_dir_time, exist_ok=True)
                os.makedirs(desc_dir_weather, exist_ok=True)
                os.makedirs(preds_dir_road, exist_ok=True)
                os.makedirs(preds_dir_time, exist_ok=True)
                os.makedirs(preds_dir_weather, exist_ok=True)

                if is_feature_inferenced:
                    if eval_ver2:
                        pred_dicts = dict_out['pred_dicts'][0]
                        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                        list_pp_bbox = []
                        list_pp_cls = []

                        for idx_pred in range(len(pred_labels)):
                            x, y, z, l, w, h, th = pred_boxes[idx_pred]
                            score = pred_scores[idx_pred]

                            if score > conf_thr:
                                cls_idx = int(np.round(pred_labels[idx_pred]))
                                cls_name = class_names[cls_idx-1]
                                list_pp_bbox.append([score, x, y, z, l, w, h, th])
                                list_pp_cls.append(cls_idx)
                            else:
                                continue
                        pp_num_bbox = len(list_pp_cls)
                        dict_out_current = dict_out
                        dict_out_current.update({
                            'pp_bbox': list_pp_bbox,
                            'pp_cls': list_pp_cls,
                            'pp_num_bbox': pp_num_bbox,
                            'pp_desc': dict_out['meta'][0]['desc']
                        })
                    else:
                        dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                else:
                    dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)

                if dict_out_current is None:
                    print('* Exception error (Pipeline): dict_item is None in validation')
                    continue

                dict_out_current = dict_datum_to_kitti(self, dict_out_current)

                if len(dict_out_current['kitti_gt']) == 0: # not eval emptry label
                    pass
                else:
                    ### Gt ###
                    for idx_label, label in enumerate(dict_out_current['kitti_gt']):
                        if idx_label == 0:
                            mode = 'w'
                        else:
                            mode = 'a'

                        with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')

                    ### Process description ###
                    with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_road + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_time + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_weather + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])

                    ### Process description ###
                    if len(dict_out_current['kitti_pred']) == 0:
                        with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                    else:
                        for idx_pred, pred in enumerate(dict_out_current['kitti_pred']):
                            if idx_pred == 0:
                                mode = 'w'
                            else:
                                mode = 'a'

                            with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')

                    str_log = idx_name + '\n'
                    with open(split_path, 'a') as f:
                        f.write(str_log)
                    with open(split_path_road, 'a') as f:
                        f.write(str_log)
                    with open(split_path_time, 'a') as f:
                        f.write(str_log)
                    with open(split_path_weather, 'a') as f:
                        f.write(str_log)

            # free memory (Killed error, checked with htop)
            if 'pointer' in dict_datum.keys():
                for dict_item in dict_datum['pointer']:
                    for k in dict_item.keys():
                        if k != 'meta':
                            dict_item[k] = None
            for temp_key in dict_datum.keys():
                dict_datum[temp_key] = None
            tqdm_bar.update(1)
        tqdm_bar.close()

        ### Validate per conf ###
        all_condition_list = ['all'] + road_cond_list + time_cond_list + weather_cond_list
        for conf_thr in list_conf_thr:
            for condition in all_condition_list:
                try:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'desc')
                    split_path = path_dir + f'/{conf_thr}/' + condition + '/val.txt'

                    dt_annos = kitti.get_label_annos(preds_dir)
                    val_ids = read_imageset_file(split_path)
                    gt_annos = kitti.get_label_annos(labels_dir, val_ids)
                    list_metrics = []
                    list_results = []
                    for idx_cls_val in self.list_val_care_idx:
                        dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                        list_metrics.append(dict_metrics)
                        list_results.append(result)
                    print('Conf thr: ', str(conf_thr), ', Condition: ', condition)
                    with open(os.path.join(path_dir, f'{conf_thr}', 'complete_results.txt'), 'a') as f:
                        for dic_metric in list_metrics:
                            print('='*50)
                            print('Cls: ', dic_metric['cls'])
                            print('IoU:', dic_metric['iou'])
                            print('BEV: ', dic_metric['bev'])
                            print('3D: ', dic_metric['3d'])
                            print('-'*50)

                            f.write('Conf thr: ' + str(conf_thr) +  ', Condition: ' + condition + '\n')
                            f.write('cls: ' + dic_metric['cls'] + '\n')
                            f.write('iou: ')
                            for iou in dic_metric['iou']:
                                f.write(str(iou) + ' ')
                            f.write('\n')
                            f.write('bev: ')
                            for bev in dic_metric['bev']:
                                f.write(str(bev) + ' ')
                            f.write('\n')
                            f.write('3d  :')
                            for det3d in dic_metric['3d']:
                                f.write(str(det3d) + ' ')
                            f.write('\n\n')
                    print('\n')
                except:
                    print('* Exception error (Pipeline): Samples for the codition are not found')

        path_check = os.path.join(path_dir, 'Conf_thr', 'complete_results.txt')
        print(f'* Check {path_check}')
        ### Validate per conf ###