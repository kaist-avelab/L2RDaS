import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import scipy.stats as stats

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, l, h = size
    new_l = l
    new_h = h
    new_w = w
    #TODO : resize according to 3D tensor
    # X : [0. 76.8] => 192 / Y : [-19.2, 19.2] => 96 / Z : [-2, 10.8] => 32
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_l - opt.fineSize))
    z = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y, z), 'flip': flip}

def get_transform(opt, params, method=Image.BICUBIC, normalize=False, modal="L"):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))

    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    # if '3d_roi' in opt.resize_or_crop:
    #     transform_list.append(transforms.Lambda(lambda img: __filter_roi(img, opt.roi_3d)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        if modal =="L":
            transform_list.append(transforms.Lambda(lambda img: __flip_pc(img, params['flip'])))
        elif modal =="R":
            transform_list.append(transforms.Lambda(lambda img: __flip_tensor(img, params['flip'])))

    # Do not tensor yet
    # transform_list += [transforms.ToTensor()]

    #Not use Normalize for base
    if normalize:
        if modal =="R":
            transform_list.append(transforms.Lambda(lambda img: __normalize_radar_power(img)))
        # transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
        #                                         (0.5, 0.5, 0.5))]
        elif modal =="L":
            transform_list.append(transforms.Lambda(lambda img: __normalize_point_cloud_power(img)))

    return transforms.Compose(transform_list)

def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(pc_data, target_width, method=Image.BICUBIC):
    ow, oh = pc_data.size
    if (ow == target_width):
        return pc_data
    w = target_width
    h = int(target_width * oh / ow)
    return pc_data.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip_pc(points, flip):
    #TODO : Now, Y flip. If I have to X Flip?
    if flip:
        points[:, 1] = -points[:, 1]
    return points

def __flip_tensor(tensor, flip):
    #TODO : Now, Y flip. If I have to X Flip?
    if flip:
        tensor=np.flip(tensor, axis=1)
    return tensor

# TO see data distribution
def visualize_power_distribution(points):
    power_values = points[:, 3]

    # 히스토그램
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(power_values, bins=50, color='blue', edgecolor='black')
    plt.title('Histogram of Power Values')
    plt.xlabel('Power')
    plt.ylabel('Frequency')

    # 박스 플롯
    plt.subplot(1, 2, 2)
    plt.boxplot(power_values, vert=False)
    plt.title('Boxplot of Power Values')
    plt.xlabel('Power')

    plt.tight_layout()
    plt.show()


def log_transform_power(points):
    transformed = points.copy()
    transformed[:, 3] = np.log1p(transformed[:, 3])
    return transformed

def sqrt_transform_power(points):
    transformed = points.copy()
    transformed[:, 3] = np.sqrt(transformed[:, 3])
    return transformed

def qq_plot(data, title):
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)
    plt.show()

def normalize_intensity(points):
    intensity = points[:, 3]
    normalized_intensity = np.log1p(intensity)  # 로그 변환 예시
    return normalized_intensity

def __normalize_point_cloud_power(points):
    if points.size == 0:
        # 빈 배열이면 그냥 그대로 리턴 (또는 0으로 채운 배열을 리턴)
        return points
    # visualize_power_distribution(points)

    # log_transformed_power = log_transform_power(points)[:, 3]
    # sqrt_transformed_power = sqrt_transform_power(points)[:, 3]

    # # Q-Q plot 그리기
    # plt.figure(figsize=(15, 5))

    # plt.subplot(1, 3, 1)
    # qq_plot(points[:,3], "Q-Q Plot of Original Power Values")

    # plt.subplot(1, 3, 2)
    # qq_plot(log_transformed_power, "Q-Q Plot of Log-Transformed Power Values")

    # plt.subplot(1, 3, 3)
    # qq_plot(sqrt_transformed_power, "Q-Q Plot of Sqrt-Transformed Power Values")

    # plt.tight_layout()
    # plt.show()

    # Assuming points is a numpy array of shape (N, 4) where the columns are [x, y, z, power]
    normalize_intensity_ = normalize_intensity(points)
    max_power = np.max(normalize_intensity_)
    min_power = np.min(normalize_intensity_)
    # points[:,3] = normalize_intensity_
    points[:, 3] = (normalize_intensity_ - min_power) / (max_power - min_power) # *255.0
    # points[:,3] = points[:,3] / float(1e+2)

    return points

def __normalize_radar_power(radar_data, norm_val=1e+13):
    # 입력 데이터의 shape을 저장
    original_shape = radar_data.shape

    # 1D array로 펼치기
    radar_data_flat = radar_data.flatten()

    # 0이 아닌 값들의 인덱스를 찾기
    non_zero_indices = np.where(radar_data_flat != 0)
    zero_indices = np.where(radar_data_flat == 0)

    # 0이 아닌 값들을 추출
    non_zero_values = radar_data_flat[non_zero_indices]

    # # 로그 스케일로 정규화
    non_zero_norm = (non_zero_values / norm_val).astype(np.float32)
    # 정규화
    # non_zero_log = non_zero_values / float(1e+13)

    # 결과를 원래 shape으로 재구성
    normalized_radar_data_flat = np.full_like(radar_data_flat,-1, dtype=np.float32)
    normalized_radar_data_flat[non_zero_indices] = non_zero_norm

    # 다시 3D 형태로 변환
    normalized_radar_data = normalized_radar_data_flat.reshape(original_shape)

    return normalized_radar_data