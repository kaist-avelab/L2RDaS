'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
* comment: sensor fusion
'''

import os
import os.path as osp
import torch
import numpy as np
import open3d as o3d
import cv2
import yaml
import matplotlib.pyplot as plt

from tqdm import tqdm
from easydict import EasyDict

from scipy.io import loadmat # from matlab

from torch.utils.data import Dataset


try:
    from utils.util_calib import *
except:
    import sys
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.util_calib import *

from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import pickle
from typing import Optional, List
import copy
from utils.bbox_trans.transform_util import center_to_corner_box2d, points_in_rbbox
from utils.bbox_trans.augmentation_util import box_collision_test

from models.GAN.models import create_model
from models.skeletons.l2r_net import VoxelGeneratorWrapper
from models.backbone_3d.vfe.mean_vfe import MeanVFE_data, MaxVFE_data
from collections import OrderedDict
from utils.spconv_utils import spconv
import util.util as util
from utils.l2r.visualizer import Visualizer
from util import html
from torch.autograd import Variable
import matplotlib.cm as cm
from skimage.metrics import structural_similarity as ssim
from torchvision.models import inception_v3
from scipy import linalg
import torch.nn.functional as F
import torch.nn as nn

from scipy.spatial import ConvexHull

#---
# Inception v3 모델 로드 및 설정
class InceptionV3FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        inception = inception_v3(pretrained=True, aux_logits=False)
        self.blocks = nn.Sequential(*list(inception.children())[:-1])  # InceptionV3의 마지막 두 층을 제거합니다.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 2D 텐서를 1D 벡터로 변환합니다.
        return x
#--

roi = [0,-16,-2,72,16,7.6]
dict_cfg = dict(
    path_data = dict(
        list_dir_kradar = ['./data/K-Radar/sequences'],
        split = ['./resources/split/train.txt', './resources/split/test.txt'],
        revised_label_v1_1 = './tools/revise_label/kradar_revised_label_v1_1',
        revised_label_v2_0 = './tools/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL',
        revised_label_v2_1 = './tools/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility',
    ),
    label = { # (consider, logit_idx, rgb, bgr)
        'calib':            True,
        'onlyR':            False,
        'consider_cls':     False,
        'consider_roi':     False,
        'remove_0_obj':     False,
        'Sedan':            [True,  1,  [0, 1, 0],       [0,255,0]],
        'Bus or Truck':     [True,  2,  [1, 0.2, 0],     [0,50,255]],
        'Motorcycle':       [False, -1, [1, 0, 0],       [0,0,255]],
        'Bicycle':          [False, -1, [1, 1, 0],       [0,255,255]],
        'Bicycle Group':    [False, -1, [0, 0.5, 1],     [0,128,255]],
        'Pedestrian':       [False, -1, [0, 0, 1],       [255,0,0]],
        'Pedestrian Group': [False, -1, [0.4, 0, 1],     [255,0,100]],
        'Label':            [False, -1, [0.5, 0.5, 0.5], [128,128,128]],
    },
    label_version = 'v2_0', # ['v1_0', 'v1_1', 'v2_0', v2_1']
    item = dict(calib=True, ldr64=True, ldr128=False, rdr=False, rdr_sparse=False, cam=True, rdr_polar_3d=True, rpcs=False),
    calib = dict(z_offset=0.7),
    cam = dict(front0=True, front1=True, left0=False, left1=False, right0=False, right1=False, rear0=False, rear1=False),
    cam_process = dict(origin=(720,1280), cropped=((127,593),(0,1280)), scaled=(256,704), dir='./output/K-Radar/undistorted_imgs'),
    cam_calib = dict(load=True, dir='./resources/cam_calib/common', dir_npy='./resources/cam_calib/T_npy'),
    ldr64 = dict(processed=False, skip_line=13, n_attr=9, inside_ldr64=True, calib=True,),
    rdr = dict(cube=False,),
    rdr_sparse = dict(processed=True, dir='./output/K-Radar/rdr_sparse',),
    rdr_polar_3d = dict(processed=True, dir='./output/K-Radar/rdr_polar_3d', in_pc100p=True),
    roi = dict(filter=False, xyz=roi, keys=['ldr64', 'rdr_sparse'], check_azimuth_for_rdr=True, azimuth_deg=[-53,53]),
    rpcs = dict(processed=True, dir='./output/K-Radar/rdr_pc', keys=['pc1p', 'pc10p']),
    portion = ['10'], # ['7', '8'],
)

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

def get_o3d_line_set_from_list_infos(list_infos, color = [0., 0., 0.], is_with_arrow=False, length_arrow=1.0, length_tips=0.4):
    line_width=0.1
    x, y, z, azi_deg, l_2, w_2, h_2 = list_infos
    theta = azi_deg
    # theta = azi_deg*np.pi/180.
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

    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Increase line thickness by adding points along the lines
    thick_line_points = []
    for start, end in lines:
        start_point = np.array(points[start])
        end_point = np.array(points[end])
        line_vector = end_point - start_point
        num_points = int(np.linalg.norm(line_vector) / line_width)
        for i in range(num_points + 1):
            thick_line_points.append(start_point + i * line_vector / num_points)

    thick_line_point_cloud = o3d.geometry.PointCloud()
    thick_line_point_cloud.points = o3d.utility.Vector3dVector(thick_line_points)
    thick_line_point_cloud.colors = o3d.utility.Vector3dVector([color for _ in thick_line_points])

    return line_set, thick_line_point_cloud

class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str, optional): The category of samples. Defaults to None.
        epoch (int, optional): Sampling epoch. Defaults to None.
        shuffle (bool): Whether to shuffle indices. Defaults to False.
        drop_reminder (bool): Drop reminder. Defaults to False.
    """

    def __init__(self,
                 sampled_list: List[dict],
                 name: Optional[str] = None,
                 epoch: Optional[int] = None,
                 shuffle: bool = True,
                 drop_reminder: bool = False) -> None:
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num: int) -> List[int]:
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self) -> None:
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num: int) -> List[dict]:
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]

class KRadarDetection_v2_1_for_INF(Dataset):
    def __init__(self, cfg=None, split='all', opt=False):
        if cfg == None:
            cfg = EasyDict(dict_cfg)
            cfg_from_yaml = False
            self.cfg=cfg
        else:
            cfg_from_yaml = True
            self.cfg=cfg.DATASET

        if opt != False:
            self.opt = opt
            self.root = opt.dataroot

        self.l2r_mode=self.cfg.get('l2r_mode', False)
        if self.l2r_mode==True:
            self.l2r_seq=set(self.cfg.l2r.ava_seq)
            self.l2r_roi=self.cfg.l2r.roi
            self.l2r_res=self.cfg.l2r.resolution

        # object sampling
        self.l2r_vis = self.cfg.Vis_mode
        self.object_sample_mode=self.cfg.get('object_sample_mode', False)
        if self.object_sample_mode==True:
            self.object_sample_path=self.cfg.object_sample.file_path
            with open(self.object_sample_path, 'rb') as fr:
                self.db_infos = pickle.load(fr)
            self.filter_by_min_points_dict=self.cfg.object_sample.filter_by_min_points
            self.db_infos = self.filter_by_min_points(self.db_infos, self.filter_by_min_points_dict)
            self.obj_rate = self.cfg.object_sample.rate
            self.sample_2d = self.cfg.object_sample.sample_2d
            self.obj_sample_groups = self.cfg.object_sample.sample_groups
            self.sampler_dict = {}
            self.group_db_infos = self.db_infos  # just use db_infos
            for k, v in self.group_db_infos.items():
                self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
            self.cat2label = {'Sedan' : 0, 'Bus or Truck' : 1}
            # self.l2r_vis = self.cfg.object_sample.Vis

        # generate tensor
        self.l2r_generate = self.cfg.object_sample.make_l2r_tensor
        if self.l2r_generate==True:
            self.l2r_path = self.cfg.object_sample.l2r_path
            self.generate_network = create_model(self.opt, cfg)

            num_point_features = self.cfg.ldr64.n_used
            self.num_point_features = num_point_features
            # point_cloud_range = np.array(self.cfg.roi.xyz)
            point_cloud_range = np.array(self.cfg.l2r.roi_xyz)
            voxel_size = self.cfg.roi.voxel_size_lidar
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
                max_num_points_per_voxel=cfg.DATASET.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
                max_num_voxels=cfg.DATASET.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['train'],
            )
            self.voxel_generator_test = VoxelGeneratorWrapper(
                vsize_xyz=voxel_size,
                coors_range_xyz=point_cloud_range,
                num_point_features=num_point_features,
                max_num_points_per_voxel=cfg.DATASET.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
                max_num_voxels=cfg.DATASET.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['test'],
            )
            # self.voxel_generator_cfg = {
            #     'vsize_xyz': voxel_size,
            #     'coors_range_xyz': point_cloud_range,
            #     'num_point_features': num_point_features,
            #     'max_num_points_per_voxel': cfg.DATASET.PRE_PROCESSING.MAX_POINTS_PER_VOXEL,
            #     'max_num_voxels': cfg.DATASET.PRE_PROCESSING.MAX_NUMBER_OF_VOXELS['test'],
            # }
            self.transform_points_to_voxels = cfg.DATASET.PRE_PROCESSING.get('TRANSFORM_POINTS_TO_VOXELS', False)

            # Build modules MaxVFE
            VFE_name = cfg.DATASET.VFE.NAME
            if VFE_name == "MaxVFE":
                self.vfe = MaxVFE_data(
                    model_cfg=cfg.DATASET.VFE,
                    num_point_features=model_info_dict['num_rawpoint_features'],
                    point_cloud_range=model_info_dict['point_cloud_range'],
                    voxel_size=model_info_dict['voxel_size'],
                    grid_size=model_info_dict['grid_size'],
                )
            elif VFE_name == "MeanVFE":
                self.vfe = MeanVFE_data(
                    model_cfg=cfg.DATASET.VFE,
                    num_point_features=model_info_dict['num_rawpoint_features'],
                    point_cloud_range=model_info_dict['point_cloud_range'],
                    voxel_size=model_info_dict['voxel_size'],
                    grid_size=model_info_dict['grid_size'],
                )
            else:
                print("No VFE module installed.")
                exit()
            model_info_dict['num_point_features'] = self.vfe.get_output_feature_dim()

            self.visualizer = Visualizer(self.opt)
            self.real_tensor_log_norm = self.cfg.object_sample.real_tensor_log_norm

        self.current_epoch = 0

        self.label = self.cfg.label

        self.sampling_set = set(self.cfg.object_sample.sampling_seq)
        self.sampling_only_seq = self.cfg.object_sample.sampling_only_seq

        # TODO : when trainig after GT sampling complete, remove_0_obj=True
        # TODO : make 3 dataset? like training_1_dataset, training_2_dataset, test_dataset ???
        # if split == 'train':
        #     self.label.remove_0_obj = False
        # elif split == 'test':
        #     self.label.remove_0_obj = True

        self.label_version = self.cfg.get('label_version', 'v2_0')
        # self.load_label_in_advance = True if self.label.remove_0_obj else False
        self.load_label_in_advance = True #if self.label.remove_0_obj else False

        self.item = self.cfg.item
        self.calib = self.cfg.calib
        self.cam = self.cfg.get('cam', None)
        self.cam_calib = self.cfg.get('cam_calib', None)
        self.ldr64 = self.cfg.ldr64
        self.rdr_sparse = self.cfg.rdr_sparse
        self.rdr_polar_3d = self.cfg.get('rdr_polar_3d', None)
        self.rpcs = self.cfg.get('rpcs', None)
        self.roi = self.cfg.roi
        self.rdr_cube = self.cfg.get('rdr_cube', None)

        for temp_key in ['cam', 'rdr_polar_3d', 'rpcs']:
            if temp_key not in self.item.keys():
                self.item[temp_key] = False

        self.portion = self.cfg.get('portion', None)

        self.data_seq2aug = {'1': 4, '2' : 72, '3' : 3, '4' : 5, '5' : 3, '6' : 12, '7' : 3, '8': 4, '9' : 2, '10':2, '11':2, '12':2, '14': 7, '15':3, '16':3, '17':3, '18':3, '19':3, '20':3}
        # Existing sparse-radar metadata is only needed when loading a saved
        # point cloud.  Fresh L2RDaS export generates that data in memory.
        if not self.l2r_generate:
            path_pkl = self.rdr_sparse.get('default_dir', None)
            if path_pkl != None:
                path_pkl=osp.join(path_pkl, f'meta_info_0.pkl')
                with open(path_pkl, 'rb') as f:
                    self.default_saved_pkl = pickle.load(f)

            path_pkl = self.rdr_sparse.get('dir', None)
            if path_pkl != None:
                path_pkl=osp.join(path_pkl, f'meta_info_0.pkl')
                with open(path_pkl, 'rb') as f:
                    self.saved_pkl_0 = pickle.load(f)

        self.saved_dict_data = {}

        self.list_dict_item = self.load_dict_item(self.cfg.path_data, split)
        if cfg_from_yaml:
            self.cfg.NUM = len(self)

        self.collate_ver = self.cfg.get('collate_fn', 'v1_0') # Post-processing

        self.arr_range, self.arr_azimuth, self.arr_elevation, \
            self.arr_doppler = self.load_physical_values(is_with_doppler=True)

        arr_r = self.arr_range
        arr_a = self.arr_azimuth
        arr_e = self.arr_elevation

        r_min = np.min(arr_r)
        r_bin = np.mean(arr_r[1:]-arr_r[:-1])
        r_max = np.max(arr_r)

        a_min = np.min(arr_a)
        a_bin = np.mean(arr_a[1:]-arr_a[:-1])
        a_max = np.max(arr_a)

        e_min = np.min(arr_e)
        e_bin = np.mean(arr_e[1:]-arr_e[:-1])
        e_max = np.max(arr_e)

        self.info_rae = [
            [r_min, r_bin, r_max],
            [a_min, a_bin, a_max],
            [e_min, e_bin, e_max]]

        self.dict_cam_calib = self.get_dict_cam_calib_from_yml() \
                            if self.cam_calib is not None else None

        shuffle_points = self.cfg.get('shuffle_points', None)
        self.shuffle_points = False if shuffle_points is None else \
                                        shuffle_points.get(split, False)

        if self.rdr_polar_3d is not None:
            if self.rdr_polar_3d.get('in_pc100p', False):
                n_r = len(self.arr_range)
                n_a = len(self.arr_azimuth)
                n_e = len(self.arr_elevation)
                rae_r = np.repeat(np.repeat((self.arr_range).copy().reshape(n_r,1,1),n_a,1),n_e,2)
                rae_a = np.repeat(np.repeat((self.arr_azimuth).copy().reshape(1,n_a,1),n_r,0),n_e,2)
                rae_e = np.repeat(np.repeat((self.arr_elevation).copy().reshape(1,1,n_e),n_r,0),n_a,1)

                # Radar polar to General polar coordinate
                rae_a = -rae_a
                rae_e = -rae_e

                # For flipped azimuth & elevation angle
                xyz_x = rae_r * np.cos(rae_e) * np.cos(rae_a)
                xyz_y = rae_r * np.cos(rae_e) * np.sin(rae_a)
                xyz_z = rae_r * np.sin(rae_e)

                self.rdr_polar_3d_xyz = np.stack((xyz_x,xyz_y,xyz_z),axis=0)
                self.get_rdr_polar_3d_in_pc100p = True
            else:
                self.get_rdr_polar_3d_in_pc100p = False


        # #--------- it is for l2r
        # ### input A (label maps)
        # dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        # self.dir_A = osp.join(opt.dataroot, opt.phase + dir_A)
        # self.A_paths = sorted(make_dataset(self.dir_A))

        # ### input B (real images)
        # if opt.isTrain or opt.use_encoded_image:
        #     dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        #     self.dir_B = osp.join(opt.dataroot, opt.phase + dir_B)
        #     self.B_paths = sorted(make_dataset(self.dir_B))

        # ### instance maps
        # if not opt.no_instance:
        #     self.dir_inst = osp.join(opt.dataroot, opt.phase + '_inst')
        #     self.inst_paths = sorted(make_dataset(self.dir_inst))

        # ### load precomputed instance-wise encoded features
        # if opt.load_features:
        #     self.dir_feat = osp.join(opt.dataroot, opt.phase + '_feat')
        #     print('----------- loading features from %s ----------' % self.dir_feat)
        #     self.feat_paths = sorted(make_dataset(self.dir_feat))

        # self.dataset_size = len(self.A_paths)
        #---------
        if self.rdr_cube is not None:
            self.is_consider_roi_rdr_cb = self.rdr_cube['is_consider_roi']
            # To make BEV -> averaging power
            self.is_count_minus_1_for_bev = self.rdr_cube['IS_COUNT_MINUS_ONE_FOR_BEV']

            # Default ROI for CB (When generating CB from matlab applying interpolation)
            self.arr_bev_none_minus_1 = None
            self.arr_z_cb = np.arange(-30, 30, 0.4)
            self.arr_y_cb = np.arange(-80, 80, 0.4)
            self.arr_x_cb = np.arange(0, 100, 0.4)

            if self.is_consider_roi_rdr_cb:
                self.consider_roi_cube(self.rdr_cube['roi'])
                if self.rdr_cube['consider_roi_order'] == 'cube -> num':
                    self.consider_roi_order = 1
                elif self.rdr_cube['consider_roi_order'] == 'num -> cube':
                    self.consider_roi_order = 2
                else:
                    raise AttributeError('Check consider roi order in cfg')
                if self.rdr_cube['bev_divide_width'] == 'bin_z':
                    self.bev_divide_with = 1
                elif self.rdr_cube['bev_divide_width'] == 'none_minus_1':
                    self.bev_divide_with = 2
                else:
                    raise AttributeError('Check consider bev divide with in cfg')

        self.split = split

        # Load mask
        self.mask = self.cfg.l2r.get('load_radar_fov_mask', None)
        if self.mask != None:
            self.mask = self.load_mask(self.mask)
        else:
            print("Need load rdr cube")

        voxel_size = self.cfg.roi.voxel_size

        #  #--for FID
        # self.model = InceptionV3FeatureExtractor()

    def load_mask(self, file_path="mask.pt"):
        """ 저장된 마스크 파일을 로드 """
        self.mask = torch.load(file_path)
        print(f"Mask loaded from {file_path}")
        return self.mask


    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def consider_roi_cube(self, roi_cart, is_reflect_to_cfg=True):
        # to get indices
        self.list_roi_idx_cb = [0, len(self.arr_z_cb)-1, \
            0, len(self.arr_y_cb)-1, 0, len(self.arr_x_cb)-1]
        idx_attr = 0
        for k, v in roi_cart.items():
            if v is not None:
                min_max = np.array(v).tolist()
                # print(min_max)
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}_cb'), min_max)
                setattr(self, f'arr_{k}_cb', arr_roi)
                self.list_roi_idx_cb[idx_attr*2] = idx_min
                self.list_roi_idx_cb[idx_attr*2+1] = idx_max
                if is_reflect_to_cfg:
                    v_new = [arr_roi[0], arr_roi[-1]]
                    v_new = np.array(v_new)
                    self.rdr_cube["roi"][k] = v_new
            idx_attr += 1

    def get_arr_in_roi(self, arr, min_max):
        min_val, max_val = min_max
        idx_min = np.argmin(abs(arr-min_val))
        idx_max = np.argmin(abs(arr-max_val))
        return arr[idx_min:idx_max+1], idx_min, idx_max

    def filter_by_min_points(self, db_infos: dict, min_gt_points_dict: dict) -> dict:
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        if info['path'].split("/")[-1].split("_")[0] in self.l2r_seq:
                            filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    ### Setup ###
    def load_dict_item(self, path_data, split):
        def get_split(split_txt, list_dict_split, val):
            f = open(split_txt, 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                seq, label = line.split(',')
                list_dict_split[int(seq)][label.rstrip('\n')] = val
        list_dict_split = [dict() for _ in range(58+1)]
        get_split(path_data.split[0], list_dict_split, 'train')
        get_split(path_data.split[1], list_dict_split, 'test')

        list_seqs_w_header = []
        for path_header in path_data.list_dir_kradar:
            list_seqs = os.listdir(path_header)
            if self.portion is None:
                list_seqs_w_header.extend([(seq, path_header) for seq in list_seqs])
            else:
                for seq in list_seqs:
                    if seq in self.portion:
                        list_seqs_w_header.append((seq, path_header))
        list_seqs_w_header = sorted(list_seqs_w_header, key=lambda x: int(x[0]))

        list_dict_item = []
        if self.l2r_mode==True: #l2r mode

            for seq, path_header in list_seqs_w_header:

                # only include seq if in l2r_seq
                if seq not in self.l2r_seq: continue

                list_labels = sorted(os.listdir(osp.join(path_header, seq, 'info_label')))
                for label in list_labels:
                    path_label_v1_0 = osp.join(path_header, seq, 'info_label', label)
                    path_label_v1_1 = osp.join(path_data.revised_label_v1_1, f'{seq}_info_label_revised', label)
                    path_label_v2_0 = osp.join(path_data.revised_label_v2_0, seq, label)
                    path_label_v2_1 = osp.join(path_data.revised_label_v2_1, seq, label)
                    path_l2r_data=path_data.l2r_path
                    dict_item = dict(
                        meta = dict(
                            header = path_header, seq = seq,
                            label_v1_0 = path_label_v1_0, label_v1_1 = path_label_v1_1,
                            label_v2_0 = path_label_v2_0, label_v2_1 = path_label_v2_1,
                            l2r_path=path_l2r_data,
                            split = list_dict_split[int(seq)][label]
                        ),
                    )
                    if self.load_label_in_advance: #for not 0 objects
                        dict_item = self.get_label(dict_item)

                        # #for advanced gt sampling (according to seq)
                        # times = self.data_seq2aug[seq]
                        # if seq in self.sampling_set:
                        #     if self.sampling_only_seq and split=='train' and self.object_sample_mode:
                        #         for i in range(times-1):
                        #             dict_item_copy = copy.deepcopy(dict_item)
                        #             dict_item_copy['meta']['doGTsample'] = True
                        #             dict_item_copy['meta']['GTsample_num'] = i
                        #             list_dict_item.append(dict_item_copy)
                        # elif split=='train':
                        #     for i in range(times-1):
                        #         dict_item_copy = copy.deepcopy(dict_item)
                        #         dict_item_copy['meta']['GTsample_num'] = i
                        #         list_dict_item.append(dict_item_copy)

                        # # for GT sampling for remove_0_obj
                        # if seq in self.sampling_set and self.sampling_only_seq and split=='train' : #and self.object_sample_mode:
                        #     dict_item_copy = copy.deepcopy(dict_item)
                        #     list_dict_item.append(dict_item_copy)
                        #     dict_item['meta']['doGTsample'] = True

                        # for GT sampling for not remove_0_obj
                        if split=='train' :
                            if self.sampling_only_seq:
                                if seq not in self.sampling_set: # 1, 6, ..
                                    if dict_item['meta']['num_obj']!=0:
                                        # continue
                                        dict_item_copy = copy.deepcopy(dict_item)
                                        dict_item_copy['meta']['GTsample_num'] = 0
                                        list_dict_item.append(dict_item_copy)

                                        # dict_item_copy = copy.deepcopy(dict_item)
                                        # dict_item_copy['meta']['GTsample_num'] = 1
                                        # list_dict_item.append(dict_item_copy)
                                    else :
                                        continue
                                    # continue
                                else: # 2 3 4 5 9 .. sampling seq
                                    if dict_item['meta']['num_obj']==0:
                                        continue
                                        # dict_item_copy = copy.deepcopy(dict_item)
                                        # dict_item_copy['meta']['doGTsample'] = True
                                        # dict_item_copy['meta']['GTsample_num'] = 0
                                        # list_dict_item.append(dict_item_copy)

                                        # dict_item_copy = copy.deepcopy(dict_item)
                                        # dict_item_copy['meta']['doGTsample'] = True
                                        # dict_item_copy['meta']['GTsample_num'] = 1
                                        # list_dict_item.append(dict_item_copy)

                                    else:
                                        # dict_item_copy = copy.deepcopy(dict_item)
                                        # dict_item_copy['meta']['GTsample_num'] = 0
                                        # list_dict_item.append(dict_item_copy)

                                        dict_item_copy = copy.deepcopy(dict_item)
                                        dict_item_copy['meta']['doGTsample'] = True
                                        dict_item_copy['meta']['GTsample_num'] = 1
                                        list_dict_item.append(dict_item_copy)

                                        # if dict_item['meta']['num_obj'] >=6:
                                        #     label_list = dict_item['meta']['label']

                                        #     sedan_count = sum(1 for label in label_list if label[0] == 'Sedan')
                                        #     bus_truck_count = sum(1 for label in label_list if label[0] == 'Bus or Truck')

                                        #     if sedan_count >= self.obj_sample_groups['Sedan'] and bus_truck_count >= self.obj_sample_groups['Bus or Truck']:
                                        #         dict_item_copy = copy.deepcopy(dict_item)
                                        #         dict_item_copy['meta']['GTsample_num'] = 0
                                        #         list_dict_item.append(dict_item_copy)
                                        #     else:
                                        #         dict_item_copy = copy.deepcopy(dict_item)
                                        #         dict_item_copy['meta']['GTsample_num'] = 0
                                        #         list_dict_item.append(dict_item_copy)

                                        #         dict_item_copy = copy.deepcopy(dict_item)
                                        #         dict_item_copy['meta']['doGTsample'] = True
                                        #         dict_item_copy['meta']['GTsample_num'] = 1
                                        #         list_dict_item.append(dict_item_copy)
                                        # else:
                                        #     dict_item_copy = copy.deepcopy(dict_item)
                                        #     dict_item_copy['meta']['GTsample_num'] = 0
                                        #     list_dict_item.append(dict_item_copy)

                                        #     dict_item_copy = copy.deepcopy(dict_item)
                                        #     dict_item_copy['meta']['doGTsample'] = True
                                        #     dict_item_copy['meta']['GTsample_num'] = 1
                                        #     list_dict_item.append(dict_item_copy)

                            else:  #train + no sampling only seq
                                list_dict_item.append(dict_item)
                        else: #test
                            list_dict_item.append(dict_item)
        else:
            for seq, path_header in list_seqs_w_header:
                list_labels = sorted(os.listdir(osp.join(path_header, seq, 'info_label')))
                for label in list_labels:
                    path_label_v1_0 = osp.join(path_header, seq, 'info_label', label)
                    path_label_v1_1 = osp.join(path_data.revised_label_v1_1, f'{seq}_info_label_revised', label)
                    path_label_v2_0 = osp.join(path_data.revised_label_v2_0, seq, label)
                    path_label_v2_1 = osp.join(path_data.revised_label_v2_1, seq, label)
                    dict_item = dict(
                        meta = dict(
                            header = path_header, seq = seq,
                            label_v1_0 = path_label_v1_0, label_v1_1 = path_label_v1_1,
                            label_v2_0 = path_label_v2_0, label_v2_1 = path_label_v2_1,
                            split = list_dict_split[int(seq)][label]
                        ),
                    )
                    if self.load_label_in_advance:
                        dict_item = self.get_label(dict_item)
                    list_dict_item.append(dict_item)

        if split == 'all':
            pass
        else:
            list_dict_item = list(filter(lambda item: item['meta']['split']==split, list_dict_item))

        # Filter unavailable frames (frames wo objects) (only)
        if self.label.remove_0_obj or split=='test':
            list_dict_item = list(filter(lambda item: item['meta']['num_obj']>0, list_dict_item))

        return list_dict_item

    def load_physical_values(self, is_in_rad=True, is_with_doppler=False):
        temp_values = loadmat('./resources/info_arr.mat')
        arr_range = temp_values['arrRange']
        if is_in_rad:
            deg2rad = np.pi/180.
            arr_azimuth = temp_values['arrAzimuth']*deg2rad
            arr_elevation = temp_values['arrElevation']*deg2rad
        else:
            arr_azimuth = temp_values['arrAzimuth']
            arr_elevation = temp_values['arrElevation']
        _, num_0 = arr_range.shape
        _, num_1 = arr_azimuth.shape
        _, num_2 = arr_elevation.shape
        arr_range = arr_range.reshape((num_0,))
        arr_azimuth = arr_azimuth.reshape((num_1,))
        arr_elevation = arr_elevation.reshape((num_2,))
        if is_with_doppler:
            arr_doppler = loadmat('./resources/arr_doppler.mat')['arr_doppler']
            _, num_3 = arr_doppler.shape
            arr_doppler = arr_doppler.reshape((num_3,))
            return arr_range, arr_azimuth, arr_elevation, arr_doppler
        else:
            return arr_range, arr_azimuth, arr_elevation

    def get_label(self, dict_item):
        meta = dict_item['meta']
        temp_key = 'label_' + self.label_version
        path_label = meta[temp_key]
        ver = self.label_version

        f = open(path_label)
        lines = f.readlines()
        f.close()
        list_tuple_objs = []
        deg2rad = np.pi/180.

        header = (lines[0]).rstrip('\n')
        try:
            temp_idx, tstamp = header.split(', ')
        except: # line breaking error for v2_0
            _, header_prime, line0 = header.split('*')
            header = '*' + header_prime
            temp_idx, tstamp = header.split(', ')
            # print('* b4: ', lines)
            lines.insert(1, '*'+line0)
            lines[0] = header
            # print('* after: ', lines)
        rdr, ldr64, camf, ldr128, camr = temp_idx.split('=')[1].split('_')
        tstamp = tstamp.split('=')[1]
        dict_idx = dict(rdr=rdr, ldr64=ldr64, camf=camf,\
                        ldr128=ldr128, camr=camr, tstamp=tstamp)
        if ver == 'v1_0':
            for line in lines[1:]:
                # print(line)
                list_vals = line.rstrip('\n').split(', ')
                if len(list_vals) != 11:
                    print('* split err in ', path_label)
                    continue
                idx_p = int(list_vals[1])
                idx_b4 = int(list_vals[2])
                cls_name = list_vals[3]
                x = float(list_vals[4])
                y = float(list_vals[5])
                z = float(list_vals[6])
                th = float(list_vals[7])*deg2rad
                l = 2*float(list_vals[8])
                w = 2*float(list_vals[9])
                h = 2*float(list_vals[10])
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p, idx_b4), 'R'))
        elif ver == 'v2_0':
            for line in lines[1:]:
                # print(line)
                list_vals = line.rstrip('\n').split(', ')
                idx_p = int(list_vals[1])
                cls_name = (list_vals[2])
                x = float(list_vals[3])
                y = float(list_vals[4])
                z = float(list_vals[5])
                th = float(list_vals[6])*deg2rad
                l = 2*float(list_vals[7])
                w = 2*float(list_vals[8])
                h = 2*float(list_vals[9])
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p), 'R'))
        elif ver == 'v2_1':
            for line in lines[1:]:
                # print(line)
                list_vals = line.rstrip('\n').split(', ')
                avail = list_vals[1]
                idx_p = int(list_vals[2])
                cls_name = (list_vals[3])
                x = float(list_vals[4])
                y = float(list_vals[5])
                z = float(list_vals[6])
                th = float(list_vals[7])*deg2rad
                l = 2*float(list_vals[8])
                w = 2*float(list_vals[9])
                h = 2*float(list_vals[10])
                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p), avail))

        header = dict_item['meta']['header']
        seq = dict_item['meta']['seq']
        path_calib = osp.join(header, seq, 'info_calib', 'calib_radar_lidar.txt')
        dict_path = dict(
            calib = path_calib,
            front = osp.join(header, seq, 'cam-front', f'cam-front_{camf}.png'),
            left = osp.join(header, seq, 'cam-left', f'cam-left_{camr}.png'),
            right = osp.join(header, seq, 'cam-right', f'cam-right_{camr}.png'),
            rear = osp.join(header, seq, 'cam-rear', f'cam-rear_{camr}.png'),
            ldr64 = osp.join(header, seq, 'os2-64', f'os2-64_{ldr64}.pcd'),
            desc = osp.join(header, seq, 'description.txt'),
        )

        onlyR = self.label.onlyR
        consider_cls = self.label.consider_cls
        if consider_cls | onlyR:
            list_temp = []
            for obj in list_tuple_objs:
                cls_name, _, _, avail = obj
                if consider_cls:
                    is_consider, _, _, _ = self.label[cls_name]
                    if not is_consider:
                        continue
                if onlyR:
                    if avail != 'R':
                        continue
                list_temp.append(obj)
            list_tuple_objs = list_temp

        dict_item['meta']['calib'] = self.get_calib_values(path_calib) if self.item.calib else None
        if self.label.calib:
            list_temp = []
            dx, dy, dz = dict_item['meta']['calib']
            for obj in list_tuple_objs:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                x = x + dx
                y = y + dy
                z = z + dz
                list_temp.append((cls_name, (x, y, z, th, l, w, h), trk, avail))
            list_tuple_objs = list_temp

        if self.label.consider_roi: # after calib
            x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
            check_azimuth_for_rdr = self.roi.check_azimuth_for_rdr
            azimuth_min, azimuth_max = self.roi.azimuth_deg
            rad2deg = 180./np.pi
            temp_list = []
            for obj in list_tuple_objs:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                azimuth = np.arctan2(y, x)*rad2deg
                if check_azimuth_for_rdr & ((azimuth < azimuth_min) | (azimuth > azimuth_max)):
                    continue
                if (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max) | (z < z_min) | (z > z_max):
                    continue
                temp_list.append(obj)
            list_tuple_objs = temp_list

        num_obj = len(list_tuple_objs)

        dict_item['meta'].update(dict(
            path=dict_path, idx=dict_idx, label=list_tuple_objs, num_obj=num_obj))

        #For GTsampling mode
        dict_item['meta']['doGTsample']=False
        return dict_item

    def get_dict_cam_calib_from_yml(self):
        dict_cam_calib = dict()
        dir_cam_calib = self.cam_calib.dir
        list_yml = os.listdir(dir_cam_calib)
        for yml_file_name in list_yml:
            key_name = yml_file_name.split('.')[0].split('_')[1]
            with open(osp.join(dir_cam_calib, yml_file_name), 'r') as yml_file:
                dict_temp = yaml.safe_load(yml_file)
            dict_cam_calib[key_name] = get_matrices_from_dict_calib(dict_temp) # img_size, intrinsics, distortion, T_ldr2cam
        return dict_cam_calib

    def get_dict_cam_calib_from_npy(self, dict_item): # from save_calibration_matrix_in_npy in util_calib.py
        dir_cam_calib = self.cam_calib.dir_npy
        list_npy = os.listdir(dir_cam_calib)

        intrinsic = []
        cam2ldr = []
        ldr2cam = []
        ldr2img = []

        for npy_file_name in list_npy:
            key_name = npy_file_name.split('.')[0].split('_')[1]
            npy_file = osp.join(dir_cam_calib, npy_file_name)
            temp = np.load(npy_file)
            if key_name == 'cam2pix':
                # temp[0,0] *= self.scale_x
                # temp[1,1] *= self.scale_y
                # temp[0,2] *= self.scale_x
                # temp[1,2] *= self.scale_y
                intrinsic.append(temp[:3, :3]) # [3, 3]
            elif key_name == 'ldr2cam':
                ldr2cam.append(temp)
                temp_inv = np.linalg.inv(temp)
                cam2ldr.append(temp_inv) #[4, 4]

        for i in range(len(intrinsic)):
            lidar2image = intrinsic[i] @ ldr2cam[i][:3, :4]
            ldr2img.append(lidar2image)

        dict_item['camera_intrinsics'] = intrinsic # [3, 3]
        dict_item['camera2lidar'] = cam2ldr # [4, 4]
        dict_item['lidar2image'] = ldr2img # [3, 4]

        return dict_item

    def get_calib_values(self, path_calib):
        f = open(path_calib, 'r')
        lines = f.readlines()
        f.close()
        list_calib = list(map(lambda x: float(x), lines[1].split(',')))
        list_values = [list_calib[1], list_calib[2], self.calib['z_offset']] # X, Y, Z
        return list_values

    def get_description(self, dict_item): # ./tools/tag_generator
        f = open(dict_item['meta']['path']['desc'])
        line = f.readline()
        road_type, capture_time, climate = line.split(',')
        dict_desc = {
            'capture_time': capture_time,
            'road_type': road_type,
            'climate': climate,
        }
        f.close()
        dict_item['meta']['desc'] = dict_desc

        return dict_item
    ### Setup ###

    ### Camera ###
    def get_camera_img(self, dict_item):
        dict_path = dict_item['meta']['path']
        if self.cam.front0 or self.cam.front1:
            img_front = cv2.imread(dict_path['front'])
            dict_item['front0'] = img_front[:,:1280,:]
            dict_item['front1'] = img_front[:,1280:,:]
        if self.cam.left0 or self.cam.left1:
            img_front = cv2.imread(dict_path['left'])
            dict_item['left0'] = img_front[:,:1280,:]
            dict_item['left1'] = img_front[:,1280:,:]
        if self.cam.right0 or self.cam.left1:
            img_front = cv2.imread(dict_path['right'])
            dict_item['right0'] = img_front[:,:1280,:]
            dict_item['right1'] = img_front[:,1280:,:]
        if self.cam.rear0 or self.cam.rear1:
            img_front = cv2.imread(dict_path['rear'])
            dict_item['rear0'] = img_front[:,:1280,:]
            dict_item['rear1'] = img_front[:,1280:,:]

        return dict_item

    def save_undistorted_camera_imgs(self, key_cam='front', root_path='./output/K-Radar/undistorted_imgs'):
        root_path0 = osp.join(root_path, key_cam+'0')
        root_path1 = osp.join(root_path, key_cam+'1')

        os.makedirs(root_path0, exist_ok=True)
        os.makedirs(root_path1, exist_ok=True)

        for seq_name in range(58):
            seq_folder0 = osp.join(root_path0, f'{seq_name+1}')
            seq_folder1 = osp.join(root_path1, f'{seq_name+1}')
            os.makedirs(seq_folder0, exist_ok=True)
            os.makedirs(seq_folder1, exist_ok=True)

        key_idx = 'camf' if key_cam == 'front' else 'camr'

        list_params0 = self.dict_cam_calib[key_cam+'0']
        list_params1 = self.dict_cam_calib[key_cam+'1']

        # for img 0
        img_size0, intrinsics0, distortion0, T_ldr2cam0 = list_params0
        ncm0, _ = cv2.getOptimalNewCameraMatrix(intrinsics0, distortion0, img_size0, alpha=0.0)
        for j in range(3):
            for i in range(3):
                intrinsics0[j,i] = ncm0[j, i]
        map_x0, map_y0 = cv2.initUndistortRectifyMap(intrinsics0, distortion0, None, ncm0, img_size0, cv2.CV_32FC1)

        # for img 1
        img_size1, intrinsics1, distortion1, T_ldr2cam1 = list_params1
        ncm1, _ = cv2.getOptimalNewCameraMatrix(intrinsics1, distortion1, img_size1, alpha=0.0)
        for j in range(3):
            for i in range(3):
                intrinsics1[j,i] = ncm1[j, i]
        map_x1, map_y1 = cv2.initUndistortRectifyMap(intrinsics1, distortion1, None, ncm1, img_size1, cv2.CV_32FC1)

        for idx_sample in tqdm(range(len(self))):
            dict_item = self.__getitem__(idx_sample)

            dict_meta = dict_item['meta']
            seq_name = dict_meta['seq']
            cam_idx = dict_meta['idx'][key_idx]

            img_temp = cv2.imread(dict_meta['path']['front'])
            img0 = img_temp[:,:1280,:]
            img1 = img_temp[:,1280:,:]

            img0_undistorted = cv2.remap(img0, map_x0, map_y0, cv2.INTER_LINEAR)
            img1_undistorted = cv2.remap(img1, map_x1, map_y1, cv2.INTER_LINEAR)

            path_img0 = osp.join(root_path0, seq_name, f'cam_{cam_idx}.png')
            path_img1 = osp.join(root_path1, seq_name, f'cam_{cam_idx}.png')

            cv2.imwrite(path_img0, img0_undistorted)
            cv2.imwrite(path_img1, img1_undistorted)
    ### Camera ###

    ### LiDAR ###
    def get_ldr64(self, dict_item):
        if self.ldr64.processed: # with attr & calib & roi
            pass # TODO
        else:
            with open(dict_item['meta']['path']['ldr64'], 'r') as f:
                lines = [line.rstrip('\n') for line in f][self.ldr64.skip_line:]
                pc_lidar = [point.split() for point in lines]
                f.close()
            pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, self.ldr64.n_attr)

            if self.ldr64.inside_ldr64:
                pc_lidar = pc_lidar[np.where(
                    (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
                    (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]

            if self.ldr64.calib:
                n_pts, _ = pc_lidar.shape
                calib_vals = np.array(dict_item['meta']['calib']).reshape(-1,3).repeat(n_pts, axis=0)
                pc_lidar[:,:3] = pc_lidar[:,:3] + calib_vals

        dict_item['ldr64'] = pc_lidar

        return dict_item

    def get_ldr64_from_path(self, path_ldr64, is_calib=True):
        with open(path_ldr64, 'r') as f:
            lines = [line.rstrip('\n') for line in f][self.ldr64.skip_line:]
            pc_lidar = [point.split() for point in lines]
            f.close()
        pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, self.ldr64.n_attr)

        if self.ldr64.inside_ldr64:
            pc_lidar = pc_lidar[np.where(
                (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
                (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]

        if self.ldr64.calib and is_calib:
            n_pts, _ = pc_lidar.shape
            calib_vals = np.array([-2.54,0.3,0.7]).reshape(-1,3).repeat(n_pts, axis=0)
            pc_lidar[:,:3] = pc_lidar[:,:3] + calib_vals

        return pc_lidar

    def get_LidarObjSample_from_path(self, path_ldr64, label_3d, dict_item, is_calib=True):
        pc_lidar = np.fromfile(path_ldr64, dtype=np.float32)
        # pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, 4)
        pc_lidar = pc_lidar.reshape(-1,4)

        if self.ldr64.inside_ldr64:
            pc_lidar = pc_lidar[np.where(
                (pc_lidar[:, 0] > 0.01) | (pc_lidar[:, 0] < -0.01) |
                (pc_lidar[:, 1] > 0.01) | (pc_lidar[:, 1] < -0.01))]

        pc_lidar[:, :3] += label_3d[:3]
        # if self.ldr64.calib and is_calib:
        #     n_pts, _ = pc_lidar.shape
        #     calib_vals = np.array(dict_item['meta']['calib']).reshape(-1,3).repeat(n_pts, axis=0)
        #     pc_lidar[:,:3] = pc_lidar[:,:3] + calib_vals

        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
        # geometries = [point_cloud]

        # # Create a visualizer object
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # for geometry in geometries:
        #     vis.add_geometry(geometry)

        # vis.poll_events()
        # vis.update_renderer()

        # # Pause to visualize
        # vis.run()

        # # Close the visualizer window
        # vis.destroy_window()

        return pc_lidar
    ### LiDAR ###

    ### 4D Radar ###
    def get_tesseract(self, dict_item):
        seq = dict_item['meta']['seq']
        rdr_idx = dict_item['meta']['idx']['rdr']
        path_tesseract = osp.join(dict_item['meta']['header'],seq,'radar_tesseract',f'tesseract_{rdr_idx}.mat')
        arr_tesseract = loadmat(path_tesseract)['arrDREA']
        arr_tesseract = np.transpose(arr_tesseract, (0, 1, 3, 2)) # DRAE
        dict_item['tesseract'] = arr_tesseract

        return dict_item

    def get_cube_polar(self, dict_item, normalizer=1e+13):
        dict_item = self.get_tesseract(dict_item)
        tesseract = dict_item['tesseract'][1:,:,:,:]/normalizer
        cube_pw = np.mean(tesseract, axis=0, keepdims=False)

        # (1) softmax (not used: overflow)
        # tesseract_exp = np.exp(tesseract)
        # tesseract_exp_sum = np.repeat(np.sum(tesseract_exp, axis=0, keepdims=True), 63, axis=0)
        # tesseract_dist = tesseract_exp/tesseract_exp_sum

        # (2) sum
        tesseract_sum = np.repeat(np.sum(tesseract, axis=0, keepdims=True), 63, axis=0)
        tesseract_dist = tesseract/tesseract_sum

        tesseract_dop = np.reshape(self.arr_doppler[1:], (63,1,1,1)).repeat(256,1).repeat(107,2).repeat(37,3)
        cube_dop = np.sum(tesseract_dist*tesseract_dop, axis=0, keepdims=False)

        dict_item['cube_pw_polar'] = cube_pw
        dict_item['cube_dop_cartesian'] = cube_dop

        return dict_item

    def get_cube(self, dict_item, is_in_log=False, mode=1):
#    def get_cube(self, path_cube, is_in_log=False, mode=0):
        '''
        * mode 0: arr_cube, mask, cnt
        * mode 1: arr_cube
        '''
        seq = dict_item['meta']['seq']
        rdr_idx = dict_item['meta']['idx']['rdr']
        path_cube = osp.join(dict_item['meta']['header'],seq,'radar_zyx_cube',f'cube_{rdr_idx}.mat')


        arr_cube = np.flip(loadmat(path_cube)['arr_zyx'], axis=0) # z-axis is flipped

        # print(arr_cube.shape)
        # print(np.count_nonzero(arr_cube==-1.))

        if (self.is_consider_roi_rdr_cb) & (self.consider_roi_order == 1):
            idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
            arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]

        # print(arr_cube.shape)

        if self.is_count_minus_1_for_bev:
            bin_z = len(self.arr_z_cb)
            if self.bev_divide_with == 1:
                bin_y = len(self.arr_y_cb)
                bin_x = len(self.arr_x_cb)
                # print(bin_z, bin_y, bin_x)
                arr_bev_none_minus_1 = np.full((bin_y, bin_x), bin_z)
            elif self.bev_divide_with == 2:
                arr_bev_none_minus_1 = bin_z-np.count_nonzero(arr_cube==-1., axis=0)
                arr_bev_none_minus_1 = np.maximum(arr_bev_none_minus_1, 1) # evade divide 0
            # print('* max: ', np.max(arr_bev_none_minus_1))
            # print('* min: ', np.min(arr_bev_none_minus_1))

        # print(arr_bev_none_minus_1.shape)

        if (self.is_consider_roi_rdr_cb) & (self.consider_roi_order == 2):
            idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
            # print(idx_z_min, idx_z_max)
            arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
            if self.is_count_minus_1_for_bev:
                arr_bev_none_minus_1 = arr_bev_none_minus_1[idx_y_min:idx_y_max+1, idx_x_min:idx_x_max+1]

        # print(arr_bev_none_minus_1.shape)

        if is_in_log:
            arr_cube[np.where(arr_cube==-1.)]= 1.
            # arr_cube = np.maximum(arr_cube, 1.) # get rid of -1 before log
            arr_cube = 10*np.log10(arr_cube)
        else:
            arr_cube = arr_cube / 1e13   #normalize with 1e13
            arr_cube = np.maximum(arr_cube, 0.)

        none_zero_mask = np.nonzero(arr_cube)

        # print(arr_cube.shape)
        # log

        if mode == 0:
            return arr_cube, none_zero_mask, arr_bev_none_minus_1
        elif mode == 1:
            dict_item['rdr_cube'] = arr_cube
            return dict_item

    def save_polar_3d(self, root_path='./output/K-Radar/rdr_polar_3d', idx_start=3500, idx_end=4605):
        for seq_name in range(58):
            seq_folder = osp.join(root_path, f'{seq_name+1}')
            os.makedirs(seq_folder, exist_ok=True)

        for idx_sample in tqdm(range(len(self))):
            if (idx_sample < idx_start) or (idx_sample > idx_end):
                continue
            try:
                # dict_item = self.__getitem__(idx_sample)
                # dict_meta = dict_item['meta']
                # seq_name = dict_meta['seq']
                # rdr_idx = dict_meta['idx']['rdr']
                # print('* seq:', seq_name, ', rdr:', rdr_idx)

                dict_item = self.__getitem__(idx_sample)

                dict_meta = dict_item['meta']
                seq_name = dict_meta['seq']
                rdr_idx = dict_meta['idx']['rdr']

                path_polar_3d = osp.join(root_path, seq_name, f'polar3d_{rdr_idx}.npy')
                if os.path.exists(path_polar_3d):
                    continue

                dict_item = self.get_cube_polar(dict_item)

                cube_pw = dict_item['cube_pw_polar']
                cube_dop = dict_item['cube_dop_cartesian']
                cube_polar = np.stack((cube_pw, cube_dop), axis=0)
                # cube_polar = cube_polar.astype(np.float32)
                # print(cube_polar.shape)

                np.save(path_polar_3d, cube_polar)

                # free memory (Killed error, checked with htop)
                for k in dict_item.keys():
                    if k != 'meta':
                        dict_item[k] = None
            except:
                seq = dict_item['meta']['seq']
                rdr_idx = dict_item['meta']['idx']['rdr']
                path_tesseract = osp.join(dict_item['meta']['header'],seq,'radar_tesseract',f'tesseract_{rdr_idx}.mat')
                print(f'* An error happens in {path_tesseract}')

    def get_rdr_sparse(self, dict_item):
        if self.rdr_sparse.processed:
            dir_rdr_sparse = self.rdr_sparse.dir
            seq = dict_item['meta']['seq']
            rdr_idx = dict_item['meta']['idx']['rdr']
            path_rdr_sparse = osp.join(dir_rdr_sparse, seq, f'sprdr_{rdr_idx}.npy')
            rdr_sparse = np.load(path_rdr_sparse)
            dict_item['rdr_sparse'] = rdr_sparse
        else: # from cube or tesseract
            rate = self.rdr_sparse.get('rate', 0.001)
            dict_item = self.get_portional_rdr_points_from_tesseract(dict_item, rate)
            dict_item['rdr_sparse'] = dict_item['rdr_pc']
        return dict_item

    def get_saved_rdr_sparse(self, dict_item):
        if self.rdr_sparse.processed:
            # dir_rdr_sparse = self.rdr_sparse.dir
            seq = dict_item['meta']['seq']
            rdr_idx = dict_item['meta']['idx']['rdr']
            doGTsample = str(dict_item['meta']['doGTsample'])
            if doGTsample=='False':
                gt_sample_num='-1'
            else:
                gt_sample_num = dict_item['meta']["GTsample_num"]

            if self.split == "test":
                dir_rdr_sparse = self.rdr_sparse.test_dir
                path_rdr_sparse = osp.join(dir_rdr_sparse, seq, f'rpc_0_{rdr_idx}_{doGTsample}.npy')
            else:
                key_name = f'{str(seq)}_{str(rdr_idx)}_{str(doGTsample)}_{str(gt_sample_num)}'

                # ----- it is bug TODO : fix if seq is sampling seq but 0 obj
                # try :
                #     key_item = copy.deepcopy(self.saved_pkl_0[key_name])
                # except KeyError:
                #     gt_sample_num = '0'
                #     doGTsample = 'True'
                #     key_name = f'{str(seq)}_{str(rdr_idx)}_{str(doGTsample)}_{str(gt_sample_num)}'
                #     key_item = copy.deepcopy(self.saved_pkl_0[key_name])
                #---------
                if doGTsample=='False':
                    dir_rdr_sparse = self.rdr_sparse.default_dir
                    key_item = copy.deepcopy(self.default_saved_pkl[key_name])
                    path_rdr_sparse = osp.join(dir_rdr_sparse, seq, f'rpc_0_{rdr_idx}_{doGTsample}_{gt_sample_num}.npy')
                else:
                    dir_rdr_sparse = self.rdr_sparse.dir
                    # try:
                    #     key_item = copy.deepcopy(self.saved_pkl_0[key_name])
                    # except KeyError:
                    #     gt_sample_num='0'
                    #     key_name = f'{str(seq)}_{str(rdr_idx)}_{str(doGTsample)}_{str(gt_sample_num)}'
                    #     key_item = copy.deepcopy(self.saved_pkl_0[key_name])

                    key_item = copy.deepcopy(self.saved_pkl_0[key_name])
                    path_rdr_sparse = osp.join(dir_rdr_sparse, seq, f'rpc_{self.current_epoch}_{rdr_idx}_{doGTsample}_{gt_sample_num}.npy')

                copied_label = copy.deepcopy(key_item['label'])
                copied_num_obj = copy.deepcopy(key_item['num_obj'])

                dict_item['meta']['label'] = copied_label
                dict_item['meta']['num_obj'] = copied_num_obj

                # path_rdr_sparse = osp.join(dir_rdr_sparse, seq, f'rpc_0_{rdr_idx}_{doGTsample}_{gt_sample_num}.npy')

            rdr_sparse = np.load(path_rdr_sparse)
            dict_item['rdr_sparse'] = rdr_sparse

        else: # from cube or tesseract
            rate = self.rdr_sparse.get('rate', 0.001)
            dict_item = self.get_portional_rdr_points_from_tesseract(dict_item, rate)
            dict_item['rdr_sparse'] = dict_item['rdr_pc']


        #Vis l2r generator & pc cloud
        if self.l2r_vis : #self.l2r_vis:
            sparse_rdr_cube = rdr_sparse
            real = False
            vis_in_sampled_sphere = False
            def plot_power_distribution(points, powers, bbox_index):
                # Power distribution plot
                plt.figure(figsize=(10, 5))
                plt.hist(powers, bins=50, color='blue', alpha=0.7)
                plt.title(f'Bounding Box {bbox_index} Power value distribution')
                plt.xlabel('Power')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.show()

            def plot_xyz_distribution(points, bbox_index):
                # XYZ distribution scatter plot
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111, projection='3d')


                scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3], cmap='jet', marker='o')
                ax.set_title(f'Bounding Box {bbox_index} XYZ coord distribtion')
                ax.set_xlabel('X-axis')
                ax.set_ylabel('Y-axis')
                ax.set_zlabel('Z-axis')

                # Add color bar to show power levels
                cbar = fig.colorbar(scatter, ax=ax, label='Power')
                plt.show()

            if 'before_num_obj' not in dict_item['meta'].keys():
                dict_item['meta']['before_num_obj'] = dict_item['meta']['num_obj']
            print("REAL =", real, "tensor data after point cloud")
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            if 'ldr64' in dict_item.keys():
                pc_lidar = dict_item['ldr64']
            else:
                # 'ldr64' 키가 없으면 빈 포인트 클라우드를 만듦
                pc_lidar = np.empty((0, 3))
            if 'ori_ldr64' in dict_item.keys() and real==True:
                pc_lidar = dict_item['ori_ldr64']
            elif real==False and 'gt_points_ldr64' in dict_item.keys():
                pc_lidar = dict_item['gt_points_ldr64']


            # pc_lidar = np.empty((0, 3))

            # pc_lidar = dict_item['ldr64']
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
            geometries=[pcd]

            for i, obj in enumerate(dict_item['meta']['label']):
                cls_name, [x,y,z,theta,l,w,h], _,_ = obj
                list_infos = [x,y,z,theta,l,w,h]
                if i < dict_item['meta']['before_num_obj']:
                    bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,0,0])
                    geometries.append(bbox_line_set)
                    geometries.append(thick_line_point_cloud)
                elif real==False:
                    bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,165/255,0])
                    geometries.append(bbox_line_set)
                    geometries.append(thick_line_point_cloud)

            if vis_in_sampled_sphere:
                list_indices = np.arange(len(sparse_rdr_cube)).tolist()
                import random
                list_indices = random.sample(list_indices, 100)
                for idx_temp in list_indices:
                    x, y, z, pw = sparse_rdr_cube[idx_temp]
                    sphere = self.create_sphere(radius=0.5, rgb=[0.,0.,0.], center=[x, y, z])
                    vis.add_geometry(sphere)
            else:
                pcd_radar = o3d.geometry.PointCloud()
                pcd_radar.points = o3d.utility.Vector3dVector(sparse_rdr_cube[:, :3])
                print(sparse_rdr_cube.shape[0])
                # colored by radar power
                # --mode : black color
                pcd_radar.paint_uniform_color([0., 0., 0.])

                # --mode 2 : colored by power
                # colors = self.color_by_power(sparse_rdr_cube)
                # pcd_radar.colors = o3d.utility.Vector3dVector(colors)
                geometries.append(pcd_radar)

            for geometry in geometries:
                vis.add_geometry(geometry)

            vis.poll_events()
            vis.update_renderer()

            vis.run()
            vis.destroy_window()


            #----for bbox power distribution
            # for i, obj in enumerate(dict_item['meta']['label']):
            #     cls_name, [x,y,z,theta,l,w,h], _,_ = obj
            #     list_infos = np.array([[x,y,z,theta,l,w,h]])
            #     if i < dict_item['meta']['before_num_obj']:
            #         filtered_points = self.remain_points_in_boxes(sparse_rdr_cube,list_infos )
            #         # bbox_point_set = get_3d_bbox_points(list_infos)
            #         # filtered_points = filter_points_in_bbox(sparse_rdr_cube, bbox_point_set)

            #         # 그래프 그리기
            #         if filtered_points.size > 0:
            #             print("original bbox point : ", list_infos)
            #             # 필터링된 포인트들의 power 값 추출
            #             powers_in_bbox = filtered_points[:, 3]
            #             plot_power_distribution(filtered_points, powers_in_bbox, i)
            #             plot_xyz_distribution(filtered_points, i)
            #     else:
            #         filtered_points = self.remain_points_in_boxes(sparse_rdr_cube,list_infos )
            #         # bbox_point_set = get_3d_bbox_points(list_infos)
            #         # filtered_points = filter_points_in_bbox(sparse_rdr_cube, bbox_point_set)

            #         # 그래프 그리기
            #         if filtered_points.size > 0:
            #             print("new bbox point : ", list_infos)
            #             # 필터링된 포인트들의 power 값 추출
            #             powers_in_bbox = filtered_points[:, 3]
            #             plot_power_distribution(filtered_points, powers_in_bbox, i)
            #             plot_xyz_distribution(filtered_points, i)

        return dict_item

    def get_portional_rdr_points_from_tesseract(self, dict_item, rate=0.001):
        dict_item = self.get_cube_polar(dict_item)

        cube_pw = dict_item['cube_pw_polar'] # Normalized with 1e+13
        cube_dop = dict_item['cube_dop_cartesian'] # Expectation w/ pw dist

        extracted_ind = np.where(cube_pw > np.quantile(cube_pw, 1-rate))
        r_ind, a_ind, e_ind = extracted_ind
        pw = cube_pw[extracted_ind]
        dop = cube_dop[extracted_ind]

        r = self.arr_range[r_ind]
        az = self.arr_azimuth[a_ind]
        el = self.arr_elevation[e_ind]

        # Radar polar to General polar coordinate
        az = -az
        el = -el

        # For flipped azimuth & elevation angle
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        dict_item['rdr_pc'] = np.stack((x,y,z,pw,dop), axis=1)

        return dict_item

    def get_rdr_polar_3d(self, dict_item):
        cfg_rdr_polar_3d = self.rdr_polar_3d

        if cfg_rdr_polar_3d.processed:
            dict_meta = dict_item['meta']
            seq_name = dict_meta['seq']
            rdr_idx = dict_meta['idx']['rdr']
            path_polar_3d = osp.join(cfg_rdr_polar_3d.dir, seq_name, f'polar3d_{rdr_idx}.npy')
            cube_polar = np.load(path_polar_3d)
        else:
            dict_item = self.get_cube_polar(dict_item)
            cube_pw = dict_item['cube_pw_polar']
            cube_dop = dict_item['cube_dop_cartesian']
            cube_polar = np.stack((cube_pw, cube_dop), axis=0)

        if self.get_rdr_polar_3d_in_pc100p:
            pc100p = np.concatenate((self.rdr_polar_3d_xyz, cube_polar), axis=0)
            n_dim, _, _, _ = pc100p.shape
            dict_item['pc100p'] = pc100p.reshape(n_dim,-1).transpose()
        else:
            dict_item['rdr_polar_3d'] = cube_polar

        return dict_item # 2, 256, 107, 37 (pw is normalized with 1e+13)

    def get_proportional_rdr_points(self, dict_item, rate=0.01, range_wise=False, with_rae_and_ind=False):
        cube_polar = dict_item['rdr_polar_3d']
        # print(cube_polar.shape)
        cube_pw = cube_polar[0,:,:,:]
        cube_dop = cube_polar[1,:,:,:]

        if range_wise:
            n_r, n_a, n_e = cube_pw.shape
            quantile_value = np.quantile(cube_pw, 1-rate, axis=(1,2))
            quantile_value = quantile_value.reshape(n_r,1,1)
            quantile_value = np.repeat(quantile_value, n_a, 1)
            quantile_value = np.repeat(quantile_value, n_e, 2)
            extracted_ind = np.where(cube_pw > quantile_value)
        else:
            extracted_ind = np.where(cube_pw > np.quantile(cube_pw, 1-rate))

        r_ind, a_ind, e_ind = extracted_ind
        pw = cube_pw[extracted_ind]
        dop = cube_dop[extracted_ind]

        r = self.arr_range[r_ind]
        az = self.arr_azimuth[a_ind]
        el = self.arr_elevation[e_ind]

        # Radar polar to General polar coordinate
        az = -az
        el = -el

        # For flipped azimuth & elevation angle
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        if with_rae_and_ind:
            rdr_pc = np.stack((x,y,z,pw,dop,r,az,el,r_ind,a_ind,e_ind), axis=1)
        else:
            rdr_pc = np.stack((x,y,z,pw,dop), axis=1)

        dict_item['rdr_pc'] = rdr_pc

        return dict_item

    def get_proportional_rdr_points_from_pc100p(self, dict_item, rate=0.01):
        pc100p = dict_item['pc100p']
        quantile_value = np.quantile(pc100p[:,3], 1-rate)
        rdr_pc = pc100p[np.where(pc100p[:,3]>quantile_value)[0],:]

        dict_item['rdr_pc'] = rdr_pc

        return dict_item

    def save_proportional_rdr_pc(self, dict_save=None, root_path='./output/K-Radar/rdr_pc'):
        if dict_save is None:
            dict_save = dict(
                folder_name = 'pc10p-all',
                rate = 0.1,
                range_wise = False,
            )

        folder_name = dict_save['folder_name']
        rate = dict_save['rate']
        range_wise = dict_save['range_wise']

        root_path = osp.join(root_path, folder_name)

        for seq_name in range(58):
            seq_folder = osp.join(root_path, f'{seq_name+1}')
            os.makedirs(seq_folder, exist_ok=True)

        for idx_sample in tqdm(range(len(self))):
            try:
                dict_item = self.__getitem__(idx_sample)

                dict_meta = dict_item['meta']
                seq_name = dict_meta['seq']
                rdr_idx = dict_meta['idx']['rdr']

                dict_item = self.get_proportional_rdr_points(dict_item, rate, range_wise, with_rae_and_ind=True)

                path_rpc = osp.join(root_path, seq_name, f'rpc_{rdr_idx}.npy')

                np.save(path_rpc, dict_item['rdr_pc'])

                # free memory (Killed error, checked with htop)
                for k in dict_item.keys():
                    if k != 'meta':
                        dict_item[k] = None
            except:
                seq = dict_item['meta']['seq']
                rdr_idx = dict_item['meta']['idx']['rdr']
                path_tesseract = osp.join(dict_item['meta']['header'],seq,'radar_tesseract',f'tesseract_{rdr_idx}.mat')
                print(f'* An error happens in {path_tesseract}')

    def get_rpcs(self, dict_item):
        if self.rpcs is None:
            return dict_item
        else:
            keys = self.rpcs.keys
            dict_meta = dict_item['meta']
            seq_name = dict_meta['seq']
            rdr_idx = dict_meta['idx']['rdr']

            for temp_key in keys:
                dict_item[temp_key] = np.load(osp.join(self.rpcs.dir, temp_key, seq_name, f'rpc_{rdr_idx}.npy'))

        return dict_item
    ### 4D Radar ###

    ### Utils ###
    def filter_roi_middle(self, dict_item):
        x_min, x_max, y_min, y_max, z_min, z_max = self.l2r_roi
        list_keys = self.roi.keys
        for temp_key in list_keys:
            if temp_key in ['rdr_sparse', 'ldr64']:
                temp_data = dict_item[temp_key]
                temp_data = temp_data[np.where(
                    (temp_data[:, 0] > x_min) & (temp_data[:, 0] < x_max) &
                    (temp_data[:, 1] > y_min) & (temp_data[:, 1] < y_max) &
                    (temp_data[:, 2] > z_min) & (temp_data[:, 2] < z_max))]
                dict_item[temp_key] = temp_data
            # elif temp_key == 'label': # moved to dict item

        # if self.label.consider_roi: # after calib
        #     x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
        #     check_azimuth_for_rdr = self.roi.check_azimuth_for_rdr
        #     azimuth_min, azimuth_max = self.roi.azimuth_deg
        #     rad2deg = 180./np.pi
        #     temp_list = []
        #     for obj in dict_item['meta']['label']:
        #         cls_name, (x, y, z, th, l, w, h), trk, avail = obj
        #         azimuth = np.arctan2(y, x)*rad2deg
        #         if check_azimuth_for_rdr & ((azimuth < azimuth_min) | (azimuth > azimuth_max)):
        #             continue
        #         if (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max) | (z < z_min) | (z > z_max):
        #             continue
        #         temp_list.append(obj)
        #     dict_item['meta']['label'] = temp_list
        #     dict_item['meta']['num_obj'] = len(temp_list)

        # -----------------Open3D로 시각화
        if self.l2r_vis :
            # all included point cloud
            print("Input lidar data, befor put in radar synthesis model")
            point_cloud_ = o3d.geometry.PointCloud()
            point_cloud_.points = o3d.utility.Vector3dVector(dict_item['ldr64'][:,:3])
            geometries = [point_cloud_]

            # geometries=[]

            for bbox in dict_item['meta']['label']:
                _, [x, y, z, theta, xl, yl, zl], _ , _= bbox
                list_infos=[x, y, z, theta, xl, yl, zl]
                bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,0,0])
                geometries.append(bbox_line_set)
                geometries.append(thick_line_point_cloud)

            # Create a visualizer object
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            for geometry in geometries:
                vis.add_geometry(geometry)

            vis.poll_events()
            vis.update_renderer()

                # Pause to visualize
            vis.run()

                # Close the visualizer window
            vis.destroy_window()
            #----------------------------

        return dict_item
    def filter_roi(self, dict_item):
        x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
        list_keys = self.roi.keys
        for temp_key in list_keys:
            if temp_key in ['rdr_sparse', 'ldr64']:
                if temp_key in dict_item.keys():
                    temp_data = dict_item[temp_key]
                    temp_data = temp_data[np.where(
                        (temp_data[:, 0] > x_min) & (temp_data[:, 0] < x_max) &
                        (temp_data[:, 1] > y_min) & (temp_data[:, 1] < y_max) &
                        (temp_data[:, 2] > z_min) & (temp_data[:, 2] < z_max))]
                    dict_item[temp_key] = temp_data
            # elif temp_key == 'label': # moved to dict item

        for temp_key in ['rdr_cube', 'generated_tensor']:
            if temp_key in dict_item.keys() and dict_item[temp_key] is not None:
                tensor_3d = dict_item[temp_key]

                x_min_idx = int(x_min / 0.4)
                x_max_idx = int(x_max / 0.4)
                y_min_idx = int(tensor_3d.shape[1] / 2 + (y_min/0.4))
                y_max_idx = int(tensor_3d.shape[1] / 2 + (y_max/0.4))
                z_min_idx = int((z_min +2.0) / 0.4)
                z_max_idx = int((z_max +2.0) / 0.4) +1 #to 24
                tensor_3d = tensor_3d[z_min_idx: z_max_idx, y_min_idx : y_max_idx, x_min_idx : x_max_idx+1] #TBD : why code error in here?
                dict_item[temp_key] = tensor_3d

        # -----------------Open3D로 시각화
        if self.l2r_vis :
            if 'before_num_obj' not in dict_item['meta'].keys():
                dict_item['meta']['before_num_obj'] = dict_item['meta']['num_obj']
            print(" After GT sampling, lidar data with roi")
            # all included point cloud
            point_cloud_ = o3d.geometry.PointCloud()

            if 'ldr64' in dict_item.keys():
                pc_lidar = dict_item['ldr64']
            else:
                # 'ldr64' 키가 없으면 빈 포인트 클라우드를 만듦
                pc_lidar = np.empty((0, 3))

            if 'gt_points_ldr64' in dict_item.keys():
                pc_lidar = dict_item['gt_points_ldr64']

            # if 'ldr64' in dict_item.keys():
            #     point_cloud_.points = o3d.utility.Vector3dVector(dict_item['ldr64'][:, :3])
            # else:
            #     # 'ldr64' 키가 없으면 빈 포인트 클라우드를 만듦
            #     point_cloud_.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            point_cloud_.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
            geometries = [point_cloud_]

            # geometries=[]

            for i, bbox in enumerate(dict_item['meta']['label']):
                _, [x, y, z, theta, xl, yl, zl], _ , _= bbox
                list_infos=[x, y, z, theta, xl, yl, zl]
                if i < dict_item['meta']['before_num_obj']:
                    bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,0,0])
                else:
                    bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,165/255,0])
                geometries.append(bbox_line_set)
                geometries.append(thick_line_point_cloud)

            # Create a visualizer object
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            for geometry in geometries:
                vis.add_geometry(geometry)

            vis.poll_events()
            vis.update_renderer()

                # Pause to visualize
            vis.run()

                # Close the visualizer window
            vis.destroy_window()
            #----------------------------

        return dict_item
    ### Utils ###

    def get_l2r(self, dict_item, index):
        seq = dict_item['meta']['seq']
        lidar_idx = dict_item['meta']['idx']['ldr64']
        rdr_idx=dict_item['meta']['idx']['rdr']

        bev_width = int(round((self.l2r_roi[1] - self.l2r_roi[0]) / self.l2r_res))
        bev_length = int(round((self.l2r_roi[3] - self.l2r_roi[2]) / self.l2r_res))
        bev_height = int(round((self.l2r_roi[5] - self.l2r_roi[4]) / self.l2r_res))
        A_size=(bev_width, bev_length, bev_height)

        ## input A (label maps)
        A=dict_item['ldr64'][:, :4]

        params = get_params(self.opt, A_size) #{'crop_pos' : (x, y), 'filp' : True/False} #get randomly crop position, flip

        if dict_item['L2RDaS_train'] == False:
            params['flip']=False

        if not self.opt.no_flip and params['flip']:
            if dict_item['meta']['num_obj']>0:
                label_list=[]
                for i, bbox in enumerate(dict_item['meta']['label']):
                    class_name, [x, y, z, theta, xl, yl, zl], tracking_num , tmp = bbox

                    # Y축 대칭: y 좌표를 반전하고 theta를 180도 회전
                    y = -y

                    # theta 조정: Y축 대칭을 고려하여 반대 방향으로 회전
                    theta = -theta
                    if theta < -np.pi:
                        theta += 2 * np.pi
                    elif theta > np.pi:
                        theta -= 2 * np.pi

                    # bbox 업데이트
                    new_bbox = (class_name, [x, y, z, theta, xl, yl, zl], tracking_num, tmp)
                    label_list.append(new_bbox)
                dict_item['meta']['label'] = label_list


        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params, modal="L", normalize=True)
            A_tensor = transform_A(A)
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0

        ### input B (real images)
        # B=dict_item['rdr_cube']
        # transform_B = get_transform(self.opt, params, modal="R")
        # B_tensor = transform_B(B)


        dict_item['ldr64']=A_tensor
        # dict_item['rdr_cube']=B_tensor

        # print("after")
        # self.vis_in_open3d(dict_item, vis_list=['ldr64', 'label'])

        return dict_item

    def plot_bev_boxes(self,bev_boxes):
        """
        Plots the BEV boxes on a 2D plane.

        Args:
            bev_boxes (np.ndarray): Array of shape (N, 4, 2) containing N boxes with 4 corners each,
                                    where each corner is represented by (x, y).
        """
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        for box in bev_boxes:
            polygon = plt.Polygon(box, edgecolor='r', facecolor='none')
            ax.add_patch(polygon)

        plt.xlim(0, 80)
        plt.ylim(-50, 50)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Bird\'s Eye View (BEV) Bounding Boxes')
        plt.grid(True)
        plt.show()

    def sample_class_v2(self, name: str, num: int,
                        gt_bboxes: List[dict], dict_item) -> List[dict]:
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)

        if gt_bboxes.shape[1] != 0:
            num_gt = gt_bboxes.shape[0]
        else:
            num_gt = 0

        if num_gt == 0:
            sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
            if self.label.calib:
                dx, dy, dz = dict_item['meta']['calib']
                sp_boxes[:, 0] += dx
                sp_boxes[:, 1] += dy
                sp_boxes[:, 2] += dz
            boxes = sp_boxes.copy()

            # Check ROI boundaries
            x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
            valid_indices = np.where(
                (sp_boxes[:, 0] >= x_min) & (sp_boxes[:, 0] <= x_max) &
                (sp_boxes[:, 1] >= y_min) & (sp_boxes[:, 1] <= y_max) &
                (sp_boxes[:, 2] >= z_min) & (sp_boxes[:, 2] <= z_max)
            )[0]
            sp_boxes = sp_boxes[valid_indices]
            sampled = [sampled[i] for i in valid_indices]

            sp_boxes_bv = center_to_corner_box2d(
                sp_boxes[:, 0:2], sp_boxes[:, 4:6], sp_boxes[:, 3])
            total_bv = sp_boxes_bv.cpu().numpy()
        else:
            #기존 scene의 gt bbox임
            gt_bboxes_bv = center_to_corner_box2d(
                gt_bboxes[:, 0:2], gt_bboxes[:, 4:6], gt_bboxes[:, 3])

            # sample bboxes
            sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
            if self.label.calib:
                dx, dy, dz = dict_item['meta']['calib']
                sp_boxes[:, 0] += dx
                sp_boxes[:, 1] += dy
                sp_boxes[:, 2] += dz
            boxes = sp_boxes.copy()

            # Check ROI boundaries
            x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
            valid_indices = np.where(
                (sp_boxes[:, 0] >= x_min) & (sp_boxes[:, 0] <= x_max) &
                (sp_boxes[:, 1] >= y_min) & (sp_boxes[:, 1] <= y_max) &
                (sp_boxes[:, 2] >= z_min) & (sp_boxes[:, 2] <= z_max)
            )[0]
            sp_boxes = sp_boxes[valid_indices]
            sampled = [sampled[i] for i in valid_indices]

            boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

            sp_boxes_new = boxes[gt_bboxes.shape[0]:]
            sp_boxes_bv = center_to_corner_box2d(
                sp_boxes_new[:, 0:2], sp_boxes_new[:, 4:6], sp_boxes_new[:, 3])
            total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)

        # self.plot_bev_boxes(total_bv) #vis
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        num_sampled = len(sampled)
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                dx, dy, dz = dict_item['meta']['calib']
                sampled[i - num_gt]['box3d_lidar'][0] += dx
                sampled[i - num_gt]['box3d_lidar'][1] += dy
                sampled[i - num_gt]['box3d_lidar'][2] += dz
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

    def remove_points_in_boxes(self, points,boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = points_in_rbbox(points[:,:3], boxes)
        points = points[np.logical_not(masks.any(-1))]

        # #v-------is--------------------
        # masked_points = copy_points[masks.any(-1)]
        # masked_point_cloud = o3d.geometry.PointCloud()
        # masked_point_cloud.points = o3d.utility.Vector3dVector(masked_points[:,:3])

        # geometries = [masked_point_cloud]

        # for bbox in boxes:
        #     [x, y, z, theta, xl, yl, zl]= bbox
        #     list_infos=[x, y, z, theta, xl, yl, zl]
        #     bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,0,0])
        #     geometries.append(bbox_line_set)
        #     geometries.append(thick_line_point_cloud)

        # # Create a visualizer object
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()

        # # Add original point cloud

        # for geometry in geometries:
        #     vis.add_geometry(geometry)

        # vis.poll_events()
        # vis.update_renderer()

        # vis.run()
        # vis.destroy_window()
        # #-------
        return points

    def remain_points_in_boxes(self, points, boxes):
        masks = points_in_rbbox(points[:, :3], boxes)
        points = points[masks.any(-1)]
        return points

    def sample_all(self,
                gt_labels: np.ndarray,
                dict_item: dict,
                img: Optional[np.ndarray] = None) -> dict:
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.
            img (np.ndarray, optional): Image array. Defaults to None.
            ground_plane (np.ndarray, optional): Ground plane information.
                Defaults to None.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels
                  of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`):
                  sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name in self.obj_sample_groups.keys():
            max_sample_num = self.obj_sample_groups[class_name]
            sampled_num = int(max_sample_num -
                              np.sum([name == class_name for name, bbox, _, _ in gt_labels]))
            sampled_num = np.round(self.obj_rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        if len(gt_labels)==0:
            avoid_coll_boxes=np.array([[]])
        else:
            avoid_coll_boxes = np.array([bbox for name, bbox, _, _ in gt_labels])

        # -----------------Open3D로 시각화
        ##### To see avoid_coll_boxes(real kradar label) is correct #######
        # # original point scenes
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(dict_item['ldr64'][:, :3])
        #  # Bounding box 시각화
        # geometries = [point_cloud]

        # for bbox in avoid_coll_boxes:
        #     [x, y, z, theta, xl, yl, zl]= bbox
        #     list_infos=[x, y, z, theta, xl, yl, zl]
        #     bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,0,0])
        #     geometries.append(bbox_line_set)
        #     geometries.append(thick_line_point_cloud)

        # # Create a visualizer object
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # for geometry in geometries:
        #     vis.add_geometry(geometry)

        # vis.poll_events()
        # vis.update_renderer()

        # # Pause to visualize
        # vis.run()

        # # Close the visualizer window
        # vis.destroy_window()
        #----------------------------

        for class_name, sampled_num in zip(self.obj_sample_groups.keys(),
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes, dict_item)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    if avoid_coll_boxes.size == 0:
                        avoid_coll_boxes = sampled_gt_box
                    else:
                        avoid_coll_boxes = np.concatenate([avoid_coll_boxes, sampled_gt_box], axis=0)



        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]

            # num_sampled = len(sampled)
            s_points_list = []
            count = 0
            data_root = '/'.join(self.object_sample_path.split("/")[:-1])
            for info in sampled:
                file_path = os.path.join(
                    data_root, 'lidars',
                    info['path'].split("/")[-1])
                s_points = self.get_LidarObjSample_from_path(file_path, info['box3d_lidar'], dict_item)
                # s_points[:, :3] +=info['box3d_lidar'][:3]

                count += 1

                s_points_list.append(s_points)
                gt_labels.append((info['name'], info['box3d_lidar'], None, 'R'))

            gt_labels_ = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.int64)

            # # -----------------Open3D로 시각화
            # # all included point cloud
            # #add point scenes
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(dict_item['ldr64'][:, :3])
            # # Bounding box 시각화
            # tmp = np.concatenate(s_points_list, axis=0)[:, :3]
            # point_cloud_ = o3d.geometry.PointCloud()
            # point_cloud_.points = o3d.utility.Vector3dVector(tmp)
            # geometries = [point_cloud_]
            # geometries.append(point_cloud)
            # # geometries=[]

            # for bbox in gt_labels:
            #     _, [x, y, z, theta, xl, yl, zl], _ , _= bbox
            #     list_infos=[x, y, z, theta, xl, yl, zl]
            #     bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,0,0])
            #     geometries.append(bbox_line_set)
            #     geometries.append(thick_line_point_cloud)

            # # Create a visualizer object
            # vis = o3d.visualization.Visualizer()
            # vis.create_window()
            # for geometry in geometries:
            #     vis.add_geometry(geometry)

            # vis.poll_events()
            # vis.update_renderer()

            # # Pause to visualize
            # vis.run()

            # # Close the visualizer window
            # vis.destroy_window()
            # #----------------------------

            ret = {
                'gt_labels_3d':
                gt_labels_,
                'gt_bboxes_3d':
                sampled_gt_bboxes,
                'points':
                # s_points_list[0].cat(s_points_list),
                np.concatenate(s_points_list, axis=0),
                'group_ids':
                np.arange(gt_labels_.shape[0],
                          gt_labels_.shape[0] + len(sampled)),
                'gt_labels':
                gt_labels
            }

        return ret
    def get_object_sampling(self, dict_item):
        #------
        #Vis original
        if self.l2r_vis:
            print("Original Lidar scene after ROI")
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            temp_data = dict_item['ldr64']
            x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
            temp_data = temp_data[np.where(
                (temp_data[:, 0] > x_min) & (temp_data[:, 0] < x_max) &
                (temp_data[:, 1] > y_min) & (temp_data[:, 1] < y_max) &
                (temp_data[:, 2] > z_min) & (temp_data[:, 2] < z_max))]
            point_cloud_ = o3d.geometry.PointCloud()
            point_cloud_.points = o3d.utility.Vector3dVector(temp_data[:,:3])
            geometries = [point_cloud_]

            for i, bbox in enumerate(dict_item['meta']['label']):
                _, [x, y, z, theta, xl, yl, zl], _ , _= bbox
                list_infos=[x, y, z, theta, xl, yl, zl]

                bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,0,0])

                geometries.append(bbox_line_set)
                geometries.append(thick_line_point_cloud)

            for geometry in geometries:
                vis.add_geometry(geometry)

            vis.run()
            vis.destroy_window()

            dict_item['ori_ldr64']=temp_data
        #-----

        gt_labels_3d = dict_item['meta']['label']

         # change to float for blending operation
        points = dict_item['ldr64']
        num_obj = dict_item['meta']['num_obj']

        #--
        dict_item['meta']['before_num_obj'] = num_obj
        #--

        if self.sample_2d:
            # TODO : Image object sampling
            # img = dict_item['img']
            # gt_bboxes_2d = dict_item['gt_bboxes']
            # # Assume for now 3D & 2D bboxes are the same
            # sampled_dict = self.db_sampler.sample_all(
            #     gt_bboxes_3d.numpy(),
            #     gt_labels_3d,
            #     gt_bboxes_2d=gt_bboxes_2d,
            #     img=img)
            pass
        else:
            sampled_dict = self.sample_all(
                gt_labels_3d,
                dict_item,
                img=None,)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            all_gt_bboxes_3d = sampled_dict['gt_labels']

            # gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
            #                               axis=0)
            # gt_bboxes_3d = gt_bboxes_3d.new_box(
            #     np.concatenate([gt_bboxes_3d.numpy(), sampled_gt_bboxes_3d]))
            gt_labels_3d=all_gt_bboxes_3d

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            # points = points.cat([sampled_points, points])
            points = np.concatenate([sampled_points, points[:,:4]], axis=0).copy()

            if self.sample_2d:
                #TODO :
                # sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                # gt_bboxes_2d = np.concatenate(
                #     [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                # input_dict['gt_bboxes'] = gt_bboxes_2d
                # input_dict['img'] = sampled_dict['img']
                pass

            num_obj = len(all_gt_bboxes_3d)
            dict_item['is_sampled']=True
        else:
            dict_item['is_sampled']=False

        # input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        # input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.int64)
        dict_item['ldr64'] = points
        dict_item['meta']['label'] = gt_labels_3d
        dict_item['meta']['num_obj'] = num_obj

        # edge case. No object sampling & no object in the scene.
        if num_obj == 0:
            # print("num_obj=0")
            return self.get_object_sampling(dict_item)


        return dict_item

    def pre_processor(self, batch_dict):
        # Shuffle (DataProcessor.shuffle_points)
        batched_ldr64 = batch_dict['ldr64']
        list_points = []
        list_voxels = []
        list_voxel_coords = []
        list_voxel_num_points = []

        # list_points.append(batched_ldr64)
        # self.voxel_generator_test = VoxelGeneratorWrapper(**self.voxel_generator_cfg)

        if self.transform_points_to_voxels:
            voxels, coordinates, num_points = self.voxel_generator_test.generate(batched_ldr64)

            list_voxels.append(voxels)
            list_voxel_coords.append(coordinates)
            list_voxel_num_points.append(num_points)

        # batched_points = np.concatenate(list_points, axis=0)
        batch_dict['points'] = batched_ldr64 # x, y, z, intensity
        batch_dict['voxels'] = torch.from_numpy(np.concatenate(list_voxels, axis=0)).cuda()
        batch_dict['voxel_coords'] = torch.from_numpy(np.concatenate(list_voxel_coords, axis=0)).cuda()
        batch_dict['voxel_num_points'] = torch.from_numpy(np.concatenate(list_voxel_num_points, axis=0)).cuda()
        #batch_dict['gt_boxes'] = batch_dict['gt_boxes'].cuda()

        return batch_dict

    def create_mask(self, data):
        data=data.unsqueeze(0).unsqueeze(0).clone()
        self.mask = (data != -1).float().cuda()

    def calculate_psnr_ssim(self, original, compressed):
        if isinstance(original, list):
            psnr_list = []
            ssim_list = []
            for i in range(len(original)):
                psnr_value, ssim_value = self.calculate_psnr_ssim(original[i], compressed[i])
                psnr_list.append(psnr_value)
                ssim_list.append(ssim_value)
            return psnr_list, ssim_list

        # Convert to numpy
        image_numpy_ori = original.cpu().float().numpy()
        image_numpy_com = compressed.cpu().float().numpy()

        # Remove the height axis by averaging over the height dimension (BEV)
        image_numpy_ori_bev = np.mean(image_numpy_ori[0], axis=0)  # shape (192, 192)
        image_numpy_com_bev = np.mean(image_numpy_com[0], axis=0)  # shape (192, 192)

        # Step 1. Apply log1p to stabilize the values
        image_numpy_ori_bev = np.log1p(image_numpy_ori_bev)
        image_numpy_com_bev = np.log1p(image_numpy_com_bev)

        # Step 2. Normalize to [0,1]
        ori_min, ori_max = image_numpy_ori_bev.min(), image_numpy_ori_bev.max()
        com_min, com_max = image_numpy_com_bev.min(), image_numpy_com_bev.max()

        if ori_max - ori_min > 1e-5:
            image_numpy_ori_bev = (image_numpy_ori_bev - ori_min) / (ori_max - ori_min)
        if com_max - com_min > 1e-5:
            image_numpy_com_bev = (image_numpy_com_bev - com_min) / (com_max - com_min)

        # Step 3. Calculate MSE
        mse = np.mean((image_numpy_ori_bev - image_numpy_com_bev) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 1.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            psnr = max(psnr, 0)  # PSNR은 0보다 작지 않게 clamping

        # Step 4. Calculate SSIM
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
            for batch in dataloader:
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

    def generate_radar_tensor(self, dict_item):
        dict_item = self.pre_processor(dict_item)
        dict_item = self.vfe.process(dict_item)

        # label=torch.tensor(dict_item['voxel_features']).float()
        label = dict_item['voxel_features'].clone().detach().float()
        image=torch.tensor([0])
        inst=torch.tensor([0]) #TBD TODO

        with torch.no_grad():
            generated = self.generate_network.inference(label, inst, image, dict_item=dict_item)

            if self.mask.device != generated.device:
                self.mask = self.mask.to(generated.device)

            generated = torch.where(self.mask.bool(), generated, torch.full_like(generated, 0.0))

        # -----cfar mode 2 : tensor roi first -> quantile
        # # Define the ROI
        # x_min, y_min, z_min, x_max, y_max, z_max = [0., -16., -2., 72., 16., 7.6]
        # grid_size = 0.4

        # # Convert ROI coordinates to array indices
        # z_min_idx = int((z_min + 2) / grid_size)
        # z_max_idx = int((z_max + 2) / grid_size)
        # y_min_idx = int((y_min + 38.4) / grid_size)
        # y_max_idx = int((y_max + 38.4) / grid_size)
        # x_min_idx = int(x_min / grid_size)
        # x_max_idx = int(x_max / grid_size)

        # # Crop the generated tensor
        # generated = generated[:, :, z_min_idx:z_max_idx, y_min_idx:y_max_idx, x_min_idx:x_max_idx]

        #--------------



        dict_item['generated_tensor'] = generated.squeeze().cpu().numpy()

        #---metric for gan
        real_image = torch.tensor(dict_item['rdr_cube']).float().unsqueeze(0).unsqueeze(0)
        psnr_value, ssim_value = self.calculate_psnr_ssim(generated.data[0], real_image[0])
        # #-----calculate FID
        # fid_value = self.calculate_fid_3d(real_image,generated.data, batch_size=1, cuda=True )
        # dict_item['meta']['metric'] = [psnr_value, ssim_value, fid_value]
        fid_value=0
        dict_item['meta']['metric'] = [psnr_value, ssim_value, fid_value]
        #----

        #--vis--
        if self.l2r_vis: #self.l2r_vis:
            print("Make BEV image of lidar input, real Radar tensor data, synthesis radar tensor data")
            batch_size=1
            voxel_features = label
            # voxel_coords = torch.tensor(dict_item['voxel_coords']).int()
            voxel_coords = dict_item['voxel_coords'].clone().detach().int()

            # spatial_shape=[32,192,192]
            spatial_shape=[128,1536,1536]

            sparse_pool_1 = spconv.SparseMaxPool3d(kernel_size=3, stride=4, padding=1)
            sparse_pool_2 = spconv.SparseMaxPool3d(kernel_size=3, stride=(1,2,2), padding=1)

            batch_indices = torch.zeros((voxel_coords.shape[0], 1), dtype=torch.int32, device=voxel_coords.device)
            voxel_coords = torch.cat([batch_indices, voxel_coords], dim=1)

            sparse_input = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords, spatial_shape=spatial_shape, batch_size=batch_size)

            sparse_pooled = sparse_pool_1(sparse_input)
            sparse_pooled = sparse_pool_2(sparse_pooled)
            input_label=sparse_pooled.dense()

            input_label = input_label[:, -1, :, :, :].unsqueeze(1)

            real_image = torch.tensor(dict_item['rdr_cube']).float().unsqueeze(0).unsqueeze(0)

            # #-----
            # # vis ori_ldr64 instead voxel_feature
            # voxel_features_ = torch.tensor(dict_item['ori_ldr64'][:, :4]).float()
            # sparse_input_ = spconv.SparseConvTensor(features=voxel_features_, indices=voxel_coords, spatial_shape=spatial_shape, batch_size=batch_size)
            # input_label_=sparse_input_.dense()
            # input_label_ = input_label_[:, -1, :, :, :].unsqueeze(1)
            # #-----

            # visuals = OrderedDict([('input_label', util.tensor2label(input_label[0], self.opt.label_nc, roi=[0, -38.4, -2, 76.8, 38.4, 10.8], dict_item=dict_item)),
            #         ('synthesized_image', util.tensor2im(generated.data[0], roi=[0, -38.4, -2, 76.8, 38.4, 10.8], dict_item=dict_item)),
            #         ('real_image', util.tensor2im(real_image[0], roi=[0, -38.4, -2, 76.8, 38.4, 10.8], dict_item=dict_item, real=True))])

            visuals = OrderedDict([('input_label', util.tensor2label(input_label[0], self.opt.label_nc, roi=self.roi.xyz, dict_item=dict_item)),
                                            ('synthesized_image', util.tensor2im(generated.data[0], roi=self.roi.xyz, dict_item=dict_item)),
                                            ('real_image', util.tensor2im(real_image[0], roi=self.roi.xyz, dict_item=dict_item, real=True))])

            path = self.cfg.Vis_path
            self.visualizer.save_images(visuals, path, dict_item)
            # self.visualizer.display_current_results(visuals, tmp_epoch, tmp_epoch)


        # # 1. real_image[0]에서 0이 아닌 값 중 min
        # real_tensor = real_image[0]
        # real_nonzero = real_tensor[real_tensor != 0]
        # real_min = real_nonzero.min()

        # # 2. generated.data[0]에서 0이 아닌 값 중 min
        # gen_tensor = generated.data[0]
        # gen_nonzero = gen_tensor[gen_tensor != 0]
        # gen_min = gen_nonzero.min()

        # # 3. 두 이미지 간의 절대 차이 계산 (device 맞추기)
        # real_tensor_gpu = real_tensor.to(gen_tensor.device)
        # abs_diff = torch.abs(real_tensor_gpu - gen_tensor)

        # # 4. 누적합과 평균 계산
        # total_diff = abs_diff.sum()
        # mean_diff = abs_diff.mean()


        # visuals = OrderedDict([('input_label', util.tensor2label(input_label[0], self.opt.label_nc, roi=self.roi.xyz, dict_item=dict_item)),
        #                         ('synthesized_image', util.tensor2im(generated.data[0]-1e6, roi=self.roi.xyz, dict_item=dict_item)),
        #                         ('real_image', util.tensor2im(real_image[0], roi=self.roi.xyz, dict_item=dict_item, real=True))])

        # path = self.cfg.Vis_path
        # self.visualizer.save_images(visuals, path, dict_item)




        return dict_item

    def plot_distribution_with_power(self, point_cloud, title):
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        power = point_cloud[:, 3]

        power_min = np.min(power)
        power_max = np.max(power)
        print(f"{title} - Power min: {power_min}, Power max: {power_max}")

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title)

        axs[0, 0].hist(x, bins=50, color='b', alpha=0.7)
        axs[0, 0].set_title('X Coordinate Distribution')

        axs[0, 1].hist(y, bins=50, color='g', alpha=0.7)
        axs[0, 1].set_title('Y Coordinate Distribution')

        axs[1, 0].hist(z, bins=50, color='r', alpha=0.7)
        axs[1, 0].set_title('Z Coordinate Distribution')

        axs[1, 1].hist(power, bins=50, color='y', alpha=0.7)
        axs[1, 1].set_title('Power Value Distribution')
        axs[1, 1].set_xlabel('Power Value')
        axs[1, 1].set_ylabel('Number of Points')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # --vis ver2. power graph. x : x coord. y : num of points, color : power value
        # x = point_cloud[:, 0]
        # y = point_cloud[:, 1]
        # z = point_cloud[:, 2]
        # power = point_cloud[:, 3]

        # power_min = np.min(power)
        # power_max = np.max(power)
        # print(f"{title} - Power min: {power_min}, Power max: {power_max}")

        # fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        # fig.suptitle(title)

        # axs[0, 0].hist(x, bins=50, color='b', alpha=0.7)
        # axs[0, 0].set_title('X Coordinate Distribution')

        # axs[0, 1].hist(y, bins=50, color='g', alpha=0.7)
        # axs[0, 1].set_title('Y Coordinate Distribution')

        # axs[1, 0].hist(z, bins=50, color='r', alpha=0.7)
        # axs[1, 0].set_title('Z Coordinate Distribution')

        # sc = axs[1, 1].scatter(x, power, c=power, cmap='viridis', alpha=0.7)
        # axs[1, 1].set_title('Power Value Distribution by X Coordinate')
        # axs[1, 1].set_xlabel('X Coordinate')
        # axs[1, 1].set_ylabel('Power Value')
        # fig.colorbar(sc, ax=axs[1, 1], label='Power Value')

        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.show()

    def color_by_power(self, sparse_rdr_cube):
        power = sparse_rdr_cube[:, 3]
        norm = plt.Normalize(vmin=np.min(power), vmax=np.max(power))
        colormap = cm.plasma
        colors = colormap(norm(power))[:, :3]  # Use only RGB channels, ignore alpha
        return colors

    ### Pre-processing for RTNH ###
    def generate_sparse_rdr_cube_for_wider_rtnh(self, dict_item, vis_for_check=False, vis_in_sampled_sphere=False, real=False, quantile_rate=None):
        norm_val = float(1e+13) # 1e+13
        grid_size = 0.4

        if quantile_rate==None:
            quantile_rate = 1.0-self.cfg.object_sample.QUANTILE_RATE #cfg_sparse_data.QUANTILE_RATE


        arr_z_cb = np.arange(self.roi.xyz[2], self.roi.xyz[5], grid_size)
        arr_y_cb = np.arange(self.roi.xyz[1], self.roi.xyz[4], grid_size)
        arr_x_cb = np.arange(self.roi.xyz[0], self.roi.xyz[3], grid_size)
        #----mode 2
        # arr_z_cb = np.arange(-2, 7.6, grid_size)
        # arr_y_cb = np.arange(-16, 16, grid_size)
        # arr_x_cb = np.arange(0, 72, grid_size)
        #-----
        z_min = np.min(arr_z_cb)
        y_min = np.min(arr_y_cb)
        x_min = np.min(arr_x_cb)

        if real==False:
            arr_cube = dict_item['generated_tensor']
            # arr_cube = dict_item['rdr_cube']
        else:
            arr_cube = dict_item['rdr_cube']
            if self.object_sample_mode_per_frame == False:
                arr_cube /= norm_val

        # arr_cube = np.flip(loadmat(path_cube)['arr_zyx'], axis=0) # z-axis is flipped
        # -- mode 3--
        arr_cube = arr_cube.astype(np.float64)
        valid_mask = arr_cube != -1
        arr_cube = np.where(valid_mask, arr_cube, 0)

        original_shape = arr_cube.shape
        radar_data_flat = arr_cube.flatten()
        non_zero_indices = np.where(radar_data_flat != 0)
        non_zero_values = radar_data_flat[non_zero_indices]

        quantile_value = np.quantile(non_zero_values, quantile_rate)

        valid_indices = non_zero_values > quantile_value
        valid_non_zero_indices = non_zero_indices[0][valid_indices]
        # Get the 3D indices from the flattened indices
        z_ind, y_ind, x_ind = np.unravel_index(valid_non_zero_indices, original_shape)
        # -----

        #---mode 1
        # z_ind, y_ind, x_ind = np.where(arr_cube > np.quantile(arr_cube, quantile_rate))
        #----

        power_val = arr_cube[z_ind, y_ind, x_ind]
        z_pc_coord = ((z_min + z_ind * grid_size) - grid_size / 2)
        y_pc_coord = ((y_min + y_ind * grid_size) - grid_size / 2)
        x_pc_coord = ((x_min + x_ind * grid_size) - grid_size / 2)

        sparse_rdr_cube = np.stack((x_pc_coord, y_pc_coord, z_pc_coord, power_val), axis=-1) # N, 4

        # #point cloud roi
        x_min, y_min, z_min, x_max, y_max, z_max = self.roi.xyz
        # sparse_rdr_cube = sparse_rdr_cube[np.where(
        #     (sparse_rdr_cube[:, 0] > x_min) & (sparse_rdr_cube[:, 0] < x_max) &
        #     (sparse_rdr_cube[:, 1] > y_min) & (sparse_rdr_cube[:, 1] < y_max) &
        #     (sparse_rdr_cube[:, 2] > z_min) & (sparse_rdr_cube[:, 2] < z_max))]

        #------
        # #Vis original
        # if vis_for_check and 'rdr_sparse' in dict_item.keys():
        #     vis = o3d.visualization.Visualizer()
        #     vis.create_window()

        #     pc_lidar = dict_item['ldr64']
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
        #     geometries=[pcd]

        #     for i, obj in enumerate(dict_item['meta']['label']):
        #         cls_name, [x,y,z,theta,l,w,h], _,_ = obj
        #         list_infos = [x,y,z,theta,l,w,h]
        #         if i < dict_item['meta']['before_num_obj']:
        #             bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,0,0])
        #         else:
        #             bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,1,0])
        #         geometries.append(bbox_line_set)
        #         geometries.append(thick_line_point_cloud)

        #     if vis_in_sampled_sphere:
        #         list_indices = np.arange(len(sparse_rdr_cube)).tolist()
        #         import random
        #         list_indices = random.sample(list_indices, 100)
        #         for idx_temp in list_indices:
        #             x, y, z, pw = sparse_rdr_cube[idx_temp]
        #             sphere = self.create_sphere(radius=0.5, rgb=[0.,0.,0.], center=[x, y, z])
        #             vis.add_geometry(sphere)
        #     else:
        #         tmp_rdr_sparse = dict_item['rdr_sparse']
        #         tmp_rdr_sparse = tmp_rdr_sparse[np.where(
        #             (tmp_rdr_sparse[:, 0] > x_min) & (tmp_rdr_sparse[:, 0] < x_max) &
        #             (tmp_rdr_sparse[:, 1] > y_min) & (tmp_rdr_sparse[:, 1] < y_max) &
        #             (tmp_rdr_sparse[:, 2] > z_min) & (tmp_rdr_sparse[:, 2] < z_max))]
        #         pcd_radar = o3d.geometry.PointCloud()
        #         pcd_radar.points = o3d.utility.Vector3dVector(tmp_rdr_sparse[:, :3])
        #         # --mode 2 : colored by power
        #         colors = self.color_by_power(tmp_rdr_sparse)
        #         pcd_radar.colors = o3d.utility.Vector3dVector(colors)
        #         geometries.append(pcd_radar)

        #     for geometry in geometries:
        #         vis.add_geometry(geometry)

        #     vis.poll_events()
        #     vis.update_renderer()

        #     vis.run()
        #     vis.destroy_window()

        #-----
        #Vis l2r generator & pc cloud
        if vis_for_check:
            def plot_power_distribution(points, powers, bbox_index):
                # Power distribution plot
                plt.figure(figsize=(10, 5))
                plt.hist(powers, bins=50, color='blue', alpha=0.7)
                plt.title(f'Bounding Box {bbox_index} Power value distribution')
                plt.xlabel('Power')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.show()

            def plot_xyz_distribution(points, bbox_index):
                # XYZ distribution scatter plot
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111, projection='3d')


                scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3], cmap='jet', marker='o')
                ax.set_title(f'Bounding Box {bbox_index} XYZ coord distribtion')
                ax.set_xlabel('X-axis')
                ax.set_ylabel('Y-axis')
                ax.set_zlabel('Z-axis')

                # Add color bar to show power levels
                cbar = fig.colorbar(scatter, ax=ax, label='Power')
                plt.show()

            if 'before_num_obj' not in dict_item['meta'].keys():
                dict_item['meta']['before_num_obj'] = dict_item['meta']['num_obj']
            print("REAL =", real, "tensor data after point cloud")
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            if 'ldr64' in dict_item.keys():
                pc_lidar = dict_item['ldr64']
            else:
                # 'ldr64' 키가 없으면 빈 포인트 클라우드를 만듦
                pc_lidar = np.empty((0, 3))
            if 'ori_ldr64' in dict_item.keys() and real==True:
                pc_lidar = dict_item['ori_ldr64']
            elif real==False and 'gt_points_ldr64' in dict_item.keys():
                pc_lidar = dict_item['gt_points_ldr64']


            pc_lidar = np.empty((0, 3))

            # pc_lidar = dict_item['ldr64']
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
            geometries=[pcd]

            for i, obj in enumerate(dict_item['meta']['label']):
                cls_name, [x,y,z,theta,l,w,h], _,_ = obj
                list_infos = [x,y,z,theta,l,w,h]
                if i < dict_item['meta']['before_num_obj']:
                    bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,0,0])
                    geometries.append(bbox_line_set)
                    geometries.append(thick_line_point_cloud)
                elif real==False:
                    bbox_line_set, thick_line_point_cloud = get_o3d_line_set_from_list_infos(list_infos, color=[1,165/255,0])
                    geometries.append(bbox_line_set)
                    geometries.append(thick_line_point_cloud)

            if vis_in_sampled_sphere:
                list_indices = np.arange(len(sparse_rdr_cube)).tolist()
                import random
                list_indices = random.sample(list_indices, 100)
                for idx_temp in list_indices:
                    x, y, z, pw = sparse_rdr_cube[idx_temp]
                    sphere = self.create_sphere(radius=0.5, rgb=[0.,0.,0.], center=[x, y, z])
                    vis.add_geometry(sphere)
            else:
                pcd_radar = o3d.geometry.PointCloud()
                pcd_radar.points = o3d.utility.Vector3dVector(sparse_rdr_cube[:, :3])
                print(sparse_rdr_cube.shape[0])
                # colored by radar power
                # --mode : black color
                pcd_radar.paint_uniform_color([0., 0., 0.])

                # --mode 2 : colored by power
                # colors = self.color_by_power(sparse_rdr_cube)
                # pcd_radar.colors = o3d.utility.Vector3dVector(colors)
                geometries.append(pcd_radar)

            for geometry in geometries:
                vis.add_geometry(geometry)

            vis.poll_events()
            vis.update_renderer()

            vis.run()
            vis.destroy_window()


            #----for bbox power distribution
            # for i, obj in enumerate(dict_item['meta']['label']):
            #     cls_name, [x,y,z,theta,l,w,h], _,_ = obj
            #     list_infos = np.array([[x,y,z,theta,l,w,h]])
            #     if i < dict_item['meta']['before_num_obj']:
            #         filtered_points = self.remain_points_in_boxes(sparse_rdr_cube,list_infos )
            #         # bbox_point_set = get_3d_bbox_points(list_infos)
            #         # filtered_points = filter_points_in_bbox(sparse_rdr_cube, bbox_point_set)

            #         # 그래프 그리기
            #         if filtered_points.size > 0:
            #             print("original bbox point : ", list_infos)
            #             # 필터링된 포인트들의 power 값 추출
            #             powers_in_bbox = filtered_points[:, 3]
            #             plot_power_distribution(filtered_points, powers_in_bbox, i)
            #             plot_xyz_distribution(filtered_points, i)
            #     else:
            #         filtered_points = self.remain_points_in_boxes(sparse_rdr_cube,list_infos )
            #         # bbox_point_set = get_3d_bbox_points(list_infos)
            #         # filtered_points = filter_points_in_bbox(sparse_rdr_cube, bbox_point_set)

            #         # 그래프 그리기
            #         if filtered_points.size > 0:
            #             print("new bbox point : ", list_infos)
            #             # 필터링된 포인트들의 power 값 추출
            #             powers_in_bbox = filtered_points[:, 3]
            #             plot_power_distribution(filtered_points, powers_in_bbox, i)
            #             plot_xyz_distribution(filtered_points, i)

        # dict_item['generated_radarPC'] = sparse_rdr_cube
        # tmp = dict_item['rdr_sparse']
        # tmp[:, 3] = np.log1p(tmp[:, 3])
        # self.plot_distribution_with_power(dict_item['rdr_sparse'], 'Distribution of dict_item["rdr_sparse"]')
        # self.plot_distribution_with_power(tmp, 'Distribution of dict_item["rdr_sparse"] log1p')
        # self.plot_distribution_with_power(sparse_rdr_cube, 'Distribution of sparse_rdr_cube')


        dict_item['rdr_sparse'] = sparse_rdr_cube

        # ##----for save radar point cloud data
        self.save_GT_rdr_pc(dict_item, real=real)

        return dict_item

    def save_GT_rdr_pc(self, dict_item = None, real=False, root_path='./output/kradar'):
        dict_save = dict(folder_name = 'gt_07_v11',)

        folder_name = dict_save['folder_name']

        if real==True:
            sub_dir = 'original'
        else:
            sub_dir = 'syn'

        root_path = osp.join(root_path, sub_dir, folder_name)

        if not os.path.exists(root_path):
            os.makedirs(root_path, exist_ok=True)

        dict_meta = dict_item['meta']
        seq_name = dict_meta['seq']
        rdr_idx = dict_meta['idx']['rdr']
        doGTsample = str(dict_meta['doGTsample'])

        path_rpc = osp.join(root_path, f'rpc_{seq_name}_{rdr_idx}.npy')

        np.save(path_rpc, dict_item['rdr_sparse'])


    ### Pre-processing for RTNH ###

    # def add_bbox_point(self, dict_item):
    #     ori_points = dict_item['ldr64']

    #     # 원래 intensity 평균 계산 (bbox 경계선 포인트를 추가할 때 사용)
    #     avg_intensity = np.mean(ori_points[:, 3]) if len(ori_points) > 0 else 0.0

    #     # 기존 포인트 개수
    #     ori_points_num = len(ori_points)

    #     # Sedan 및 Bus/Truck의 One-hot Mask 초기화
    #     sedan_mask = np.zeros((ori_points_num, 1), dtype=np.float32)
    #     bus_mask = np.zeros((ori_points_num, 1), dtype=np.float32)
    #     bbox_edge_mask = np.zeros((ori_points_num, 1), dtype=np.float32)

    #     # Bounding Box 라벨이 존재하는 경우
    #     if dict_item['meta']['num_obj'] > 0:
    #         sedan_list = []
    #         bus_list = []
    #         bbox_point_list = []

    #         for bbox in dict_item['meta']['label']:
    #             class_name, [x, y, z, theta, xl, yl, zl], _, _ = bbox
    #             bbox_info = [x, y, z, theta, xl, yl, zl]

    #             if class_name == "Sedan":
    #                 sedan_list.append(bbox_info)
    #             elif class_name == "Bus or Truck":
    #                 bus_list.append(bbox_info)
    #             else:
    #                 print("Find New class! Must rewrite the code:", class_name)
    #                 exit()

    #             # BBox 경계선 포인트 생성
    #             bbox_edge_points = get_points_from_line(bbox_info)
    #             bbox_point_list.append(bbox_edge_points)

    #         # NumPy 배열 변환
    #         sedan_array = np.array(sedan_list) if len(sedan_list) > 0 else np.empty((0, 7))
    #         bus_array = np.array(bus_list) if len(bus_list) > 0 else np.empty((0, 7))

    #         # BBox 내부 포인트 찾기
    #         if sedan_array.size > 0:
    #             sedan_in_bbox = points_in_rbbox(ori_points[:, :3], sedan_array)
    #             sedan_mask[sedan_in_bbox.any(-1)] = 1.0  # Sedan 내부 포인트 표시

    #         if bus_array.size > 0:
    #             bus_in_bbox = points_in_rbbox(ori_points[:, :3], bus_array)
    #             bus_mask[bus_in_bbox.any(-1)] = 1.0  # Bus or Truck 내부 포인트 표시

    #         # BBox 경계 포인트 추가
    #         if len(bbox_point_list) > 0:
    #             bbox_points_concat = np.concatenate(bbox_point_list, axis=0)  # (M, 3) 형태
    #             num_bbox_points = bbox_points_concat.shape[0]

    #             # 추가된 bbox 경계 포인트의 intensity는 평균 intensity 값으로 설정
    #             bbox_points_extended = np.zeros((num_bbox_points, 7), dtype=np.float32)
    #             bbox_points_extended[:, :3] = bbox_points_concat  # (x, y, z)
    #             bbox_points_extended[:, 3] = avg_intensity  # 평균 intensity 값
    #             bbox_points_extended[:, 6] = 1.0  # BBox 경계선 포인트 표시 (bbox_edge_mask = 1)

    #             # 기존 ori_points 확장하여 새로운 포인트 추가
    #             ori_points_extended = np.hstack([ori_points, sedan_mask, bus_mask, bbox_edge_mask])  # 기존 포인트 확장
    #             ori_points_extended = np.vstack([ori_points_extended, bbox_points_extended])  # bbox 포인트 추가

    #         else:
    #             # bbox 포인트가 없을 경우 기존 포인트만 확장
    #             ori_points_extended = np.hstack([ori_points, sedan_mask, bus_mask, bbox_edge_mask])

    #     else:
    #         # 객체가 없는 경우에도 (x, y, z, intensity, sedan_mask, bus_mask, bbox_edge_mask) 형태 유지
    #         ori_points_extended = np.hstack([ori_points, sedan_mask, bus_mask, bbox_edge_mask])

    #     # 업데이트된 포인트를 dict_item에 저장
    #     dict_item['ldr64'] = ori_points_extended
    #     # visualize_bbox_points(dict_item)
    #     return dict_item

    def make_gaussian_sphere(self, center_xyz, radius, grid=0.4, sigma_div=3.0):
        """
        center_xyz : (3,)  – 구의 중심  (cx, cy, cz)
        radius     : float – 구 반지름 (m)
        grid       : float – 격자 간격 (m)
        sigma_div  : float – sigma = radius / sigma_div
        ---------------------------------------------------------
        return pts  : (M, 4) – [x, y, z, g]  (g: Gaussian value)
        """
        cx, cy, cz = center_xyz
        # sigma 설정: 반지름이 3σ(혹은 5σ) 정도가 되도록
        sigma = radius / sigma_div
        # 격자 범위 정의
        rng = np.arange(-radius, radius + 1e-4, grid)
        xs, ys, zs = np.meshgrid(rng, rng, rng, indexing="ij")
        coords = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)          # (K,3) 오프셋
        d2 = np.sum(coords ** 2, axis=-1)                                # 거리^2
        mask = d2 <= radius ** 2                                         # 구 내부
        coords = coords[mask]
        d2 = d2[mask]
        # Gaussian 값
        g = np.exp(-d2 / (2 * sigma ** 2))
        pts = np.hstack([coords + center_xyz, g[:, None]])               # (M,4)
        return pts

    def add_bbox_point(self, dict_item):
        ori_pts = dict_item["ldr64"]          # (N,4) – x,y,z,I
        N = len(ori_pts)
        avg_I = np.mean(ori_pts[:, 3]) if N else 0.0

        # 1. 채널 확장 (sedan, bus, edge)
        ori_pts_ext = np.hstack([
            ori_pts,                          # (x, y, z, I)
            np.zeros((N, 3), dtype=np.float32)  # (sedan, bus, edge)
        ])  # shape (N, 7)

        # 2. 좌표 → 키 매핑 (0.1m resolution)
        ori_coords = np.round(ori_pts_ext[:, :3], 1)
        # ori_keys = np.ascontiguousarray(ori_coords).view([('', np.float32)] * 3).reshape(-1)
        key_to_idx = {tuple(coord): idx for idx, coord in enumerate(ori_coords)}  # ✅ 수정됨

        # 3. 누적할 새로운 포인트들
        new_sphere_pts = []
        new_edge_pts = []
        new_sphere_dict = {}  # key: (x, y, z) → point[7]

        if dict_item["meta"]["num_obj"] > 0:
            for bbox in dict_item["meta"]["label"]:
                cls_name, [x, y, z, theta, xl, yl, zl], _, _ = bbox
                center = np.array([x, y, z], dtype=np.float32)
                bbox_info = [x, y, z, theta, xl, yl, zl]

                # Gaussian sphere
                if cls_name == "Sedan":
                    ch_idx = 4
                    sphere_pts = self.make_gaussian_sphere(center, radius=3.0, grid=0.4)
                elif cls_name == "Bus or Truck":
                    ch_idx = 5
                    sphere_pts = self.make_gaussian_sphere(center, radius=5.0, grid=0.4)
                else:
                    print("Find new class! Must rewrite the code:", cls_name)
                    exit()

                for px, py, pz, g in sphere_pts:
                    key = (round(px, 1), round(py, 1), round(pz, 1))
                    if key in key_to_idx:
                        idx = key_to_idx[key]
                        ori_pts_ext[idx, ch_idx] = max(ori_pts_ext[idx, ch_idx], g)
                    else:
                        if key not in new_sphere_dict:
                            pt = np.zeros((7,), dtype=np.float32)
                            pt[:3] = [px, py, pz]
                            pt[3] = avg_I
                            pt[ch_idx] = g
                            new_sphere_dict[key] = pt
                        else:
                            new_sphere_dict[key][ch_idx] = max(new_sphere_dict[key][ch_idx], g)

                # BBox edge points
                edge_pts = get_points_from_line(bbox_info)
                edge_pts = np.array(edge_pts)
                if edge_pts is not None and len(edge_pts) > 0:
                    # edge_coords = np.round(edge_pts, 1)
                    # edge_keys = np.ascontiguousarray(edge_coords).view([('', np.float32)] * 3).reshape(-1)
                    for i in range(edge_pts.shape[0]):
                        key = tuple(np.round(edge_pts[i], 1))
                        if key in key_to_idx:
                            idx = key_to_idx[key]
                            ori_pts_ext[idx, 6] = 1.0
                        elif key in new_sphere_dict:
                            new_sphere_dict[key][6] = 1.0  # ✅ 이미 Gaussian Sphere에 추가된 포인트면 edge 채널만 ON
                        else:
                            pt = np.zeros((7,), dtype=np.float32)
                            pt[:3] = edge_pts[i]
                            pt[3] = avg_I
                            pt[6] = 1.0
                            new_edge_pts.append(pt)

        # Step 5: 병합
        final_pts = ori_pts_ext
        if new_sphere_dict:
            new_sphere_pts = np.vstack(list(new_sphere_dict.values()))
            final_pts = np.vstack([final_pts, new_sphere_pts])
        if new_edge_pts:
            final_pts = np.vstack([final_pts, np.vstack(new_edge_pts)])

        # 저장
        dict_item["ldr64"] = final_pts
        return dict_item

    def __len__(self):
        return len(self.list_dict_item)

    def __getitem__(self, idx):
         # if sampling_only_seq mode : True, using GT sampling in sampling_seq
        # idx=idx+90
        dict_item = copy.deepcopy(self.list_dict_item[idx])

        if self.sampling_only_seq :
            self.object_sample_mode_per_frame = dict_item['meta']['doGTsample']

        # object sampling
        #synthesized
        if self.split == 'train': #train
            dict_item = self.get_label(dict_item) if not self.load_label_in_advance else dict_item
            dict_item = self.get_ldr64(dict_item) if self.item['ldr64'] else dict_item
            dict_item = self.get_camera_img(dict_item) if self.item['cam'] else dict_item
            dict_item = self.get_cube(dict_item) if self.item['rdr_cube'] else dict_item
            dict_item = self.get_description(dict_item)

            dict_item['L2RDaS_train'] = not self.l2r_generate

            dict_item = self.filter_roi_middle(dict_item) if self.roi.filter else dict_item
            dict_item = self.get_object_sampling(dict_item)

            dict_item = self.get_l2r(dict_item, idx)
            # self.vis_in_open3d(dict_item, vis_list=['ldr64'])
            dict_item = self.add_bbox_point(dict_item)
            # self.vis_in_open3d(dict_item, vis_list=['ldr64'])
            dict_item = self.generate_radar_tensor(dict_item)
            dict_item = self.filter_roi(dict_item) if self.roi.filter else dict_item
            if self.l2r_vis:
                dict_item = self.generate_sparse_rdr_cube_for_wider_rtnh(dict_item, vis_for_check=True, vis_in_sampled_sphere=False, real=True)
                dict_item = self.generate_sparse_rdr_cube_for_wider_rtnh(dict_item, vis_for_check=True, vis_in_sampled_sphere=False, real=False)
            else:
                dict_item = self.generate_sparse_rdr_cube_for_wider_rtnh(dict_item, vis_for_check=False, vis_in_sampled_sphere=False, real=False)

        # self.vis_in_open3d(dict_item, vis_list=['rdr_sparse', 'label'])
        # #real
        # dict_item = self.get_label(dict_item) if not self.load_label_in_advance else dict_item
        # dict_item = self.get_ldr64(dict_item) if self.item['ldr64'] else dict_item
        # dict_item = self.get_cube(dict_item) if self.item['rdr_cube'] else dict_item
        # dict_item = self.get_description(dict_item)

        # dict_item = self.filter_roi(dict_item) if self.roi.filter else dict_item
        # if self.l2r_vis:
        #     dict_item = self.generate_sparse_rdr_cube_for_wider_rtnh(dict_item, vis_for_check=True, vis_in_sampled_sphere=False, real=True)
        #     # dict_item = self.generate_sparse_rdr_cube_for_wider_rtnh(dict_item, vis_for_check=True, vis_in_sampled_sphere=False)
        # else:
        #     dict_item = self.generate_sparse_rdr_cube_for_wider_rtnh(dict_item, vis_for_check=False, vis_in_sampled_sphere=False, real=True)

        # # dict_item = self.generate_sparse_rdr_cube_for_wider_rtnh(dict_item, vis_for_check=True, vis_in_sampled_sphere=False, real=True, quantile_rate=1.0-0.1)
        # # dict_item = self.generate_sparse_rdr_cube_for_wider_rtnh(dict_item, vis_for_check=True, vis_in_sampled_sphere=False, real=True, quantile_rate=1.0-0.05)
        # # dict_item = self.generate_sparse_rdr_cube_for_wider_rtnh(dict_item, vis_for_check=True, vis_in_sampled_sphere=False, real=True, quantile_rate=1.0-0.01)
        # # dict_item = self.generate_sparse_rdr_cube_for_wider_rtnh(dict_item, vis_for_check=True, vis_in_sampled_sphere=False, real=True, quantile_rate=1.0-0.005)

        # self.vis_in_open3d(dict_item, vis_list=['rdr_sparse', 'label'])

        return dict_item

    ### Vis ###
    def create_cylinder_mesh(self, radius, p0, p1, color=[1, 0, 0]):
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(np.array(p1)-np.array(p0)))
        cylinder.paint_uniform_color(color)
        frame = np.array(p1) - np.array(p0)
        frame /= np.linalg.norm(frame)
        R = o3d.geometry.get_rotation_matrix_from_xyz((np.arccos(frame[2]), np.arctan2(-frame[0], frame[1]), 0))
        cylinder.rotate(R, center=[0, 0, 0])
        cylinder.translate((np.array(p0) + np.array(p1)) / 2)
        return cylinder

    def draw_3d_box_in_cylinder(self, vis, center, theta, l, w, h, color=[1, 0, 0], radius=0.1, in_cylinder=True):
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0,              0,             1]])
        corners = np.array([[l/2, w/2, h/2], [l/2, w/2, -h/2], [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
                            [-l/2, w/2, h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]])
        corners_rotated = np.dot(corners, R.T) + center
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                [0, 4], [1, 5], [2, 6], [3, 7]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_rotated)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
        if in_cylinder:
            for line in lines:
                cylinder = self.create_cylinder_mesh(radius, corners_rotated[line[0]], corners_rotated[line[1]], color)
                vis.add_geometry(cylinder)
        else:
            vis.add_geometry(line_set)

    def create_sphere(self, radius=0.2, resolution=30, rgb=[0., 0., 0.], center=[0., 0., 0.]):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
        color = np.array(rgb)
        mesh_sphere.vertex_colors = o3d.utility.Vector3dVector([color for _ in range(len(mesh_sphere.vertices))])
        x, y, z = center
        transform = np.identity(4)
        transform[0, 3] = x
        transform[1, 3] = y
        transform[2, 3] = z
        mesh_sphere.transform(transform)
        return mesh_sphere

    def vis_in_open3d(self, dict_item, vis_list=['rdr_sparse', 'ldr64', 'label']):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        if 'ldr64' in vis_list:
            pc_lidar = dict_item['ldr64']
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:,:3])
            vis.add_geometry(pcd)

        if 'rdr_sparse' in vis_list:
            rdr_sparse = dict_item['rdr_sparse']
            pcd_rdr = o3d.geometry.PointCloud()
            pcd_rdr.points = o3d.utility.Vector3dVector(rdr_sparse[:,:3])
            pcd_rdr.paint_uniform_color([0.,0.,0.])
            vis.add_geometry(pcd_rdr)

        if 'rdr_pc' in vis_list:
            rdr_sparse = dict_item['rdr_pc']
            pcd_rdr = o3d.geometry.PointCloud()
            pcd_rdr.points = o3d.utility.Vector3dVector(rdr_sparse[:,:3])
            pcd_rdr.paint_uniform_color([0.,0.,0.])
            vis.add_geometry(pcd_rdr)

        if 'label' in vis_list:
            label = dict_item['meta']['label']
            for obj in label:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                consider, logit_idx, rgb, bgr = self.label[cls_name]
                if consider:
                    self.draw_3d_box_in_cylinder(vis, (x, y, z), th, l, w, h, color=rgb, radius=0.05)
        vis.run()
        vis.destroy_window()
    ### Vis ###

    ### Distribution ###
    def get_distribution_of_label(self, consider_avail=True):
        dict_label = self.label.copy()
        dict_label.pop('calib')
        dict_label.pop('onlyR')
        dict_label.pop('Label')
        dict_label.pop('consider_cls')
        dict_label.pop('consider_roi')
        dict_label.pop('remove_0_obj')

        dict_for_dist = dict()
        dict_for_value = dict()
        dict_for_min_xyz = dict()
        dict_for_max_xyz = dict()
        for obj_name in dict_label.keys():
            dict_for_dist[obj_name] = 0
            dict_for_value[obj_name] = [0., 0., 0.]
            dict_for_min_xyz[obj_name] = [10000., 10000., 10000.]
            dict_for_max_xyz[obj_name] = [-10000., -10000., -10000.]

        if consider_avail:
            dict_avail = dict()
            list_avails = ['R', 'L', 'L1']
            for avail in list_avails:
                dict_temp = dict()
                for obj_name in dict_label.keys():
                    dict_temp[obj_name] = 0
                dict_avail[avail] = dict_temp

        for dict_item in tqdm(self.list_dict_item):
            dict_item = self.get_label(dict_item)
            for obj in dict_item['meta']['label']:
                cls_name, (x, y, z, th, l, w, h), trk, avail = obj
                dict_for_dist[cls_name] += 1
                dict_for_value[cls_name][0] += l
                dict_for_value[cls_name][1] += w
                dict_for_value[cls_name][2] += h
                dict_for_min_xyz[cls_name][0] = min(dict_for_min_xyz[cls_name][0], x)
                dict_for_max_xyz[cls_name][0] = max(dict_for_max_xyz[cls_name][0], x)
                dict_for_min_xyz[cls_name][1] = min(dict_for_min_xyz[cls_name][1], y)
                dict_for_max_xyz[cls_name][1] = max(dict_for_max_xyz[cls_name][1], y)
                dict_for_min_xyz[cls_name][2] = min(dict_for_min_xyz[cls_name][2], z)
                dict_for_max_xyz[cls_name][2] = max(dict_for_max_xyz[cls_name][2], z)

                # if x<0:
                #     print(x)

                try:
                    if consider_avail:
                        dict_avail[avail][cls_name] += 1
                except:
                    print(dict_item['meta']['label_v2_1'])

        for obj_name in dict_for_dist.keys():
            n_obj = dict_for_dist[obj_name]
            l, w, h = dict_for_value[obj_name]
            min_x, min_y, min_z = dict_for_min_xyz[obj_name]
            max_x, max_y, max_z = dict_for_max_xyz[obj_name]
            print('* # of ', obj_name, ': ', n_obj)
            divider = np.maximum(n_obj, 1)
            print('* lwh of ', obj_name, ': ', l/divider, ', ', w/divider, ', ', h/divider)
            print('* min xyz of ', obj_name, ': ', min_x, ', ', min_y, ', ', min_z)
            print('* max xyz of ', obj_name, ': ', max_x, ', ', max_y, ', ', max_z)

        if consider_avail:
            for avail in list_avails:
                print('-'*30, avail, '-'*30)
                for obj_name in dict_avail[avail].keys():
                    print('* # of ', obj_name, ': ', dict_avail[avail][obj_name])
    ### Distribution ###

    ### Collate ###
    def collate_fn(self, list_batch):
        if None in list_batch:
            print('* Exception error (Dataset): collate fn 0')
            return None

        if self.collate_ver == 'v1_0':
            dict_batch = dict()

            list_keys = list_batch[0].keys()
            for k in list_keys:
                dict_batch[k] = []
            dict_batch['label'] = []
            dict_batch['num_objs'] = []

            for batch_idx, dict_item in enumerate(list_batch):
                for k, v in dict_item.items():
                    if k == 'meta':
                        dict_batch['meta'].append(v)
                        list_objs = []

                        for tuple_obj in dict_item['meta']['label']:
                            cls_name, vals, trk_id, _ = tuple_obj
                            _, logit_idx, _, _ = self.label[cls_name]
                            list_objs.append((cls_name, logit_idx, vals, trk_id))
                        dict_batch['label'].append(list_objs)
                        dict_batch['num_objs'].append(dict_item['meta']['num_obj'])
                    elif k in ['rdr_sparse', 'ldr64']:
                        dict_batch[k].append(torch.from_numpy(dict_item[k]).float())
            dict_batch['batch_size'] = batch_idx+1

            for k in list_keys:
                if k in ['rdr_sparse', 'ldr64']:
                    batch_indices = []
                    for batch_idx, pc in enumerate(dict_batch[k]):
                        batch_indices.append(torch.full((len(pc),), batch_idx))

                    dict_batch[k] = torch.cat(dict_batch[k], dim=0)
                    dict_batch['batch_indices_'+k] = torch.cat(batch_indices)

        elif self.collate_ver == 'v2_0': # gt_boxes (B, M, 8)
            dict_batch = dict()

            list_keys = list_batch[0].keys()
            for k in list_keys:
                dict_batch[k] = []
            dict_batch['label'] = []
            dict_batch['num_objs'] = []
            dict_batch['gt_boxes'] = []
            # dict_batch['metrics'] = []

            if self.object_sample_mode: #object_sample_mode --> train rtnh with gtsampling --> no use ldr64
                list_sparse_keys = ['rdr_sparse', 'pc100p', 'pc30p', 'pc20p', 'pc15p', 'pc10p', 'pc5p', 'pc1p'] # rpcs
            else: #object sample_mode:false --> train gan --> need ldr64
                list_sparse_keys = ['rdr_sparse', 'ldr64', 'pc100p', 'pc30p', 'pc20p', 'pc15p', 'pc10p', 'pc5p', 'pc1p', 'rdr_cube'] # rpcs

            max_objs = 0 # for gt_boxes (M)
            for batch_idx, dict_item in enumerate(list_batch):
                for k, v in dict_item.items():
                    if k == 'meta':
                        dict_batch['meta'].append(v)
                        list_objs = []
                        list_gt_boxes = []
                        for tuple_obj in dict_item['meta']['label']:
                            cls_name, vals, trk_id, _ = tuple_obj
                            _, logit_idx, _, _ = self.label[cls_name]
                            list_objs.append((cls_name, logit_idx, vals, trk_id))
                            x, y, z, th, l, w, h = vals
                            list_gt_boxes.append([x, y, z, l, w, h, th, logit_idx])
                        dict_batch['label'].append(list_objs)
                        dict_batch['num_objs'].append(dict_item['meta']['num_obj'])
                        dict_batch['gt_boxes'].append(list_gt_boxes)
                        # dict_batch['metrics'].append(dict_item['meta']['metric'])
                        max_objs = max(max_objs, dict_item['meta']['num_obj'])
                    elif k in list_sparse_keys:
                        temp_points = dict_item[k]
                        if self.shuffle_points:
                            shuffle_idx = np.random.permutation(temp_points.shape[0])
                            temp_points = temp_points[shuffle_idx]
                        dict_batch[k].append(torch.from_numpy(temp_points.copy()).float())
            dict_batch['batch_size'] = batch_idx+1

            batch_size = dict_batch['batch_size']
            if self.cfg.label.remove_0_obj or self.object_sample_mode:
                gt_boxes = np.zeros((batch_size, max_objs, 8))
                for batch_idx in range(batch_size):
                    gt_box = np.array(dict_batch['gt_boxes'][batch_idx])
                    gt_boxes[batch_idx,:dict_batch['num_objs'][batch_idx],:] = gt_box
                dict_batch['gt_boxes'] = torch.tensor(gt_boxes, dtype=torch.float32)

            for k in list_keys:
                if k in list_sparse_keys:
                    batch_indices = []
                    for batch_idx, pc in enumerate(dict_batch[k]):
                        batch_indices.append(torch.full((len(pc),), batch_idx))

                    dict_batch[k] = torch.cat(dict_batch[k], dim=0)
                    dict_batch['batch_indices_'+k] = torch.cat(batch_indices)

        dict_batch['pointer'] = list_batch # to release memory

        return dict_batch
    ### Collate ###

if __name__ == '__main__':
    kradar_detection = KRadarDetection_v2_1(split='all')
    print(len(kradar_detection)) # 34994 for all

    dict_item = kradar_detection[0]
    dict_item = kradar_detection.get_dict_cam_calib_from_npy(dict_item)

    print(dict_item['camera_intrinsics']) # [3, 3]
    print(dict_item['camera2lidar']) # [4, 4]
    print(dict_item['lidar2image']) # [3, 4]

    print(kradar_detection.dict_cam_calib.keys())
    print(kradar_detection.dict_cam_calib['front0'])
    print(kradar_detection.dict_cam_calib['front1'])

    ### Camera calibration ###
    # key_name = 'front0' # ['front0', 'front1', 'left0', 'left1', 'right0', 'right1', 'rear0', 'rear1']
    # img = dict_item[key_name]
    # ldr64 = kradar_detection.get_ldr64_from_path(dict_item['meta']['path']['ldr64'], is_calib=False) # calibration X
    # show_projected_point_cloud(img, ldr64, kradar_detection.dict_cam_calib[key_name], undistort=True)
    # save_calibration_matrix_in_npy(key_name, kradar_detection.dict_cam_calib[key_name], undistort=True)
    ### Camera calibration ###

    ### Save rdr polar 3d ###
    # kradar_detection.save_polar_3d()
    ### Save rdr polar 3d ###

    ### Range-wise proportional rdr points ###
    # dict_item = kradar_detection.get_proportional_rdr_points(dict_item) # pc100p=False
    # dict_item = kradar_detection.get_proportional_rdr_points_from_pc100p(dict_item)
    ### Range-wise proportional rdr points ###

    # print(dict_item['rdr_sparse'].shape)
    # print(dict_item['rdr_pc'].shape)

    ### Vis ###
    # kradar_detection.vis_in_open3d(dict_item, ['ldr64', 'label', 'rdr_pc']) # 'rdr_pc', 'rdr_sparse'
    ### Vis ###

    ### Save rdr pc ###
    # kradar_detection.save_proportional_rdr_pc()
    ### Save rdr pc ###

    ### Save undistorted img ###
    # kradar_detection.save_undistorted_camera_imgs()
    ### Save undistorted img ###

    # kradar_detection.get_distribution_of_label()
