from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os
from matplotlib import cm
from utils.util_geometry import *
import cv2
import matplotlib.pyplot as plt
import pickle

def lidar_to_bev_v2(lidar_points, bboxes, dict_item):
    if isinstance(dict_item['meta'], list):
        x_roi=(0, 76.8)
        y_roi=(-38.4, 38.4)
        z_roi=(-2, 10.8)
        res=0.4
        arr_y_cb = np.arange(-38.4, 38.4, 0.4)
        arr_x_cb = np.arange(0, 76.8, 0.4)
    else:
        x_roi=(0, 76.8)
        y_roi=(-38.4, 38.4)
        z_roi=(-2, 10.8)
        res=0.4
        arr_y_cb = np.arange(-38.4, 38.4, 0.4)
        arr_x_cb = np.arange(0, 76.8, 0.4)
        # x_roi=(0, 72.)
        # y_roi=(-16., 16.)
        # z_roi=(-2, 7.6)
        # res=0.4
        # arr_y_cb = np.arange(-16., 16., 0.4)
        # arr_x_cb = np.arange(0, 72., 0.4)

    # BEV의 너비와 높이를 계산
    bev_width = int((x_roi[1] - x_roi[0]) / res)+1
    bev_height = int((y_roi[1] - y_roi[0]) / res)+1
    bev_z=int((z_roi[1]-z_roi[0])/res)

    # BEV 생성
    bev_map = np.full((bev_height, bev_width, 3),255, dtype=np.uint8)


    for point in lidar_points:
        x, y, z, _ = point
        # if x_roi[0] <= x <= x_roi[1] and y_roi[0] <= y <= y_roi[1] and z_roi[0] <= z <= z_roi[1]:
        # BEV 좌표 계산
        ix = int((x - x_roi[0]) / res)
        iy = int((y - y_roi[0]) / res)
        iz = int((z - z_roi[0]) / res)
        if 0 <= ix < bev_width and 0 <= iy < bev_height:
            color = cm.jet((iz) / (bev_z))[:3]  # jet colormap 사용
            bev_map[iy, ix] = (255 * np.array(color)).astype(np.uint8)

    # 객체 bbox 시각화
    for idx, bbox in enumerate(bboxes):
        _, [x, y, z, theta, xl, yl, zl], _, _ = bbox
        obj3d = Object3D(x, y, z, xl, yl, zl, theta)

        idx_x = np.argmin(np.abs(arr_x_cb - x))
        idx_y = np.argmin(np.abs(arr_y_cb - y))
        bev_map[idx_y, idx_x] = (0, 0, 0)  # 객체 중심 좌표를 검정색으로 표시

        # bbox 선 그리기
        pts = [obj3d.corners[0, :], obj3d.corners[2, :], obj3d.corners[6, :], obj3d.corners[4, :]]
        for i, pt in enumerate(pts):
            start_idx = np.argmin(np.abs(arr_x_cb - pts[i][0])), np.argmin(np.abs(arr_y_cb - pts[i][1]))
            end_idx = np.argmin(np.abs(arr_x_cb - pts[(i + 1) % len(pts)][0])), np.argmin(
                np.abs(arr_y_cb - pts[(i + 1) % len(pts)][1]))

            if isinstance(dict_item['meta'], list):
                tmp = dict_item['meta'][0]['num_obj']
            else:
                # tmp = dict_item['meta']['before_num_obj']
                tmp = dict_item['meta'].get('before_num_obj', dict_item['meta']['num_obj'])
            if idx < tmp:
                cv2.line(bev_map, start_idx, end_idx, (255, 0, 0), 1)  # bbox 선 그리기
            else:
                cv2.line(bev_map, start_idx, end_idx, (255, 255, 0), 1)  # bbox 선 그리기
    return bev_map

def radar_to_image(radar_data, bboxes, dict_item, non_zero_indices=None, real=False):
    # a=copy_bbox_data(radar_data, bboxes, dict_item, non_zero_indices=non_zero_indices, real=real, save_path="bbox_data")
    # radar_data/=255.0
    # radar_data = paste_bbox_data(radar_data, bboxes, dict_item, non_zero_indices=non_zero_indices, real=real, save_path="bbox_data" )
    if isinstance(dict_item['meta'], list):
        arr_y_cb = np.arange(-38.4, 38.4, 0.4)
        arr_x_cb = np.arange(0, 76.8, 0.4)
    else:
        arr_y_cb = np.arange(-38.4, 38.4, 0.4)
        arr_x_cb = np.arange(0, 76.8, 0.4)

    non_zero_values = radar_data[non_zero_indices]
    non_zero_values = 10*np.log10(non_zero_values*(1e+13)) #        tensor_3d = (tensor_3d*(1e+13))

    # 각 값에 대한 색상 매핑
    colormap = cm.get_cmap('jet')  # 파란색에서 빨간색으로 변경되는 colormap 선택
    norm = plt.Normalize(vmin=np.min(non_zero_values), vmax=np.max(non_zero_values))
    colored_values = colormap(norm(non_zero_values))

    # radar 데이터에서 0인 부분을 검은색으로 처리
    if radar_data.shape[0] == 80:
        colored_image = np.zeros((80, 180, 4), dtype=np.float32)
    elif radar_data.shape[0] == 192:
        colored_image = np.zeros((192, 192, 4), dtype=np.float32)                 ######~!!!!!!!!!!########(200, 248, 4) ==> radar_data shape(200, 248) + 4
    colored_image[non_zero_indices[0], non_zero_indices[1], :] = colored_values

    image = Image.fromarray((colored_image * 255).astype(np.uint8))

    # 객체 bbox 시각화
    image_np = np.array(image)  # 이미지를 numpy 배열로 변환

    # 객체 bbox 시각화
    for idx, bbox in enumerate(bboxes):
        _, [x, y, z, theta, xl, yl, zl], _, _ = bbox
        obj3d = Object3D(x, y, z, xl, yl, zl, theta)

        idx_x = np.argmin(np.abs(arr_x_cb - x))
        idx_y = np.argmin(np.abs(arr_y_cb - y))
        # image_np[idx_y, idx_x] = (0, 0, 0, 255)  # 객체 중심 좌표를 검정색으로 표시

        # bbox 선 그리기
        pts = [obj3d.corners[0, :], obj3d.corners[2, :], obj3d.corners[6, :], obj3d.corners[4, :]]
        for i, pt in enumerate(pts):
            start_idx = np.argmin(np.abs(arr_x_cb - pts[i][0])), np.argmin(np.abs(arr_y_cb - pts[i][1]))
            end_idx = np.argmin(np.abs(arr_x_cb - pts[(i + 1) % len(pts)][0])), np.argmin(
                np.abs(arr_y_cb - pts[(i + 1) % len(pts)][1]))

            if isinstance(dict_item['meta'], list):
                tmp = dict_item['meta'][0]['num_obj']
            else:
                # tmp = dict_item['meta']['before_num_obj']
                tmp = dict_item['meta'].get('before_num_obj', dict_item['meta']['num_obj'])
            if idx<tmp:
                cv2.line(image_np, start_idx, end_idx, (255, 0, 0, 255), 1)  # bbox 선 그리기
            if real==False and idx>=tmp:
                cv2.line(image_np, start_idx, end_idx, (255, 165, 0, 255), 1)

    # a=copy_bbox_data(radar_data, bboxes, dict_item, non_zero_indices=non_zero_indices, real=real, save_path="bbox_data")
    return image_np
    # 이미지 크기 조정(added)
    bev_height=192
    bev_width=192
    new_size = (bev_height * 2, bev_width * 2)  # 이미지 크기를 2배로 조정
    image_np_resized = cv2.resize(image_np, new_size[::-1], interpolation=cv2.INTER_NEAREST)

    return Image.fromarray(image_np_resized), zero_indices

def copy_bbox_data(radar_data, bboxes, dict_item, non_zero_indices=None, real=False, save_path="bbox_data"):
    # 복사할 bbox 내부 데이터 저장 리스트
    copied_data = []

    if isinstance(dict_item['meta'], list):
        arr_y_cb = np.arange(-38.4, 38.4, 0.4)
        arr_x_cb = np.arange(0, 76.8, 0.4)
    else:
        arr_y_cb = np.arange(-16., 16., 0.4)
        arr_x_cb = np.arange(0, 72., 0.4)

    # 저장할 폴더 생성
    os.makedirs(save_path, exist_ok=True)

    for idx, bbox in enumerate(bboxes):
        _, [x, y, z, theta, xl, yl, zl], _, _ = bbox

        # bbox 좌표 계산
        x_min = np.argmin(np.abs(arr_x_cb - (x - xl / 2)))
        x_max = np.argmin(np.abs(arr_x_cb - (x + xl / 2)))
        y_min = np.argmin(np.abs(arr_y_cb - (y - yl / 2)))
        y_max = np.argmin(np.abs(arr_y_cb - (y + yl / 2)))

        # bbox 내부의 원본 데이터 복사
        bbox_indices = (slice(y_min, y_max), slice(x_min, x_max))
        bbox_data = radar_data[y_min:y_max, x_min:x_max]
        copied_data.append(bbox_data)

        # bbox 좌표 및 데이터 저장 (pickle 사용)
        with open(os.path.join(save_path, f"bbox_indices_{idx}.pkl"), "wb") as f:
            pickle.dump((x_min, x_max, y_min, y_max), f)
        np.save(os.path.join(save_path, f"bbox_data_{idx}.npy"), bbox_data)

    return copied_data

def paste_bbox_data(radar_data, bboxes, dict_item, non_zero_indices=None, real=False, save_path="bbox_data"):
    # 저장된 bbox 데이터를 로드하고 원본 데이터에 적용
    modified_radar_data = radar_data.copy()

    if isinstance(dict_item['meta'], list):
        arr_y_cb = np.arange(-38.4, 38.4, 0.4)
        arr_x_cb = np.arange(0, 76.8, 0.4)
    else:
        arr_y_cb = np.arange(-16., 16., 0.4)
        arr_x_cb = np.arange(0, 72., 0.4)

    for idx, bbox in enumerate(bboxes):
        _, [x, y, z, theta, xl, yl, zl], _, _ = bbox

        if isinstance(dict_item['meta'], list):
            tmp = dict_item['meta'][0]['num_obj']
        else:
            # tmp = dict_item['meta']['before_num_obj']
            tmp = dict_item['meta'].get('before_num_obj', dict_item['meta']['num_obj'])

        if idx >= tmp:
            # bbox 좌표 계산
            x_min = np.argmin(np.abs(arr_x_cb - (x - xl / 2)))
            x_max = np.argmin(np.abs(arr_x_cb - (x + xl / 2)))
            y_min = np.argmin(np.abs(arr_y_cb - (y - yl / 2)))
            y_max = np.argmin(np.abs(arr_y_cb - (y + yl / 2)))

            # 저장된 bbox 데이터 불러오기
            bbox_data = np.load(os.path.join(save_path, f"bbox_data_{idx}.npy"))
            bbox_h, bbox_w = bbox_data.shape[:2]
            paste_h = y_max - y_min
            paste_w = x_max - x_min

            # 크기 비교 후 붙여넣기
            if bbox_h > paste_h:
                start_y = (bbox_h - paste_h) // 2
                bbox_data = bbox_data[start_y:start_y + paste_h, :]
            if bbox_w > paste_w:
                start_x = (bbox_w - paste_w) // 2
                bbox_data = bbox_data[:, start_x:start_x + paste_w]

            # small size
            if bbox_h < paste_h:
                modified_radar_data[((y_min+y_max)//2)-(bbox_h//2):((y_min+y_max)//2)+(bbox_h-(bbox_h//2)), x_min:x_min + bbox_data.shape[1]] = bbox_data
                return modified_radar_data
            # if bbox_w < paste_w:
            #     start_x = (bbox_w - paste_w) // 2
            #     bbox_data = bbox_data[:, start_x:start_x + paste_w]

            modified_radar_data[y_min:y_min + bbox_data.shape[0], x_min:x_min + bbox_data.shape[1]] = bbox_data

    return modified_radar_data

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, modal="R", roi=None, dict_item=None, real=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize, modal=modal, roi=roi, dict_item=dict_item))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()

    if modal == "L":
        x_min, y_min, z_min, x_max, y_max, z_max = roi
        lpc = dict_item['ldr64'][:,:4]
        lpc = lpc[np.where(
            (lpc[:, 0] > x_min) & (lpc[:, 0] < x_max) &
            (lpc[:, 1] > y_min) & (lpc[:, 1] < y_max) &
            (lpc[:, 2] > z_min) & (lpc[:, 2] < z_max))]
        if isinstance(dict_item['meta'], list):
            image_numpy = lidar_to_bev_v2(lpc, dict_item['meta'][0]['label'], dict_item)
        else:
            image_numpy = lidar_to_bev_v2(lpc, dict_item['meta']['label'], dict_item)
    else:
        #---add
        tensor_3d=image_numpy[0]

        if roi is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = roi
            x_min_idx = int(x_min / 0.4)
            x_max_idx = int(x_max / 0.4)
            y_min_idx = int(tensor_3d.shape[1] / 2 + (y_min/0.4))
            y_max_idx = int(tensor_3d.shape[1] / 2 + (y_max/0.4))
            z_min_idx = int((z_min +2.0) / 0.4)
            z_max_idx = int((z_max +2.0) / 0.4) +1 #to 24
            tensor_3d = tensor_3d[z_min_idx: z_max_idx, y_min_idx : y_max_idx, x_min_idx : x_max_idx] #TBD : Why +1?

        # tensor_3d = (tensor_3d+1) /2.0 * 255.0
        tensor_3d = tensor_3d.astype(np.float64)
        # image_numpy = np.clip(tensor_3d, 0, 255)

        # Create a mask where values are not -1
        valid_mask = tensor_3d != -1
        tensor_3d = np.where(valid_mask, tensor_3d, 0)

        # Calculate the mean along the z-axis (depth axis)
        image_numpy = np.mean(tensor_3d, axis=0)

        # 데이터에서 0이 아닌 값들만 추출
        non_zero_indices = np.where(image_numpy != 0)
        zero_indices=np.where(image_numpy ==0 )

        if isinstance(dict_item['meta'], list):
            image_numpy = radar_to_image(image_numpy, dict_item['meta'][0]['label'], dict_item, non_zero_indices=non_zero_indices, real=real)
        else:
            image_numpy = radar_to_image(image_numpy, dict_item['meta']['label'], dict_item, non_zero_indices=non_zero_indices, real=real)

    # 이미지 크기 조정(added)

    new_size = (int((y_max-y_min) * 4), int((x_max-x_min) * 4))  # 이미지 크기를 2배로 조정
    image_numpy = cv2.resize(image_numpy, new_size[::-1], interpolation=cv2.INTER_LINEAR)

    image_numpy = cv2.flip(image_numpy, 0)

    return image_numpy

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, roi=None, dict_item=None):
    if n_label == 0:
        return tensor2im(label_tensor, imtype, modal="L", roi=roi, dict_item=dict_item)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)],
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
