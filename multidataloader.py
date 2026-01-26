# coding=gbk
from PIL import Image
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import os

class PairedMultiScaleCropFlip:
    def __init__(self, crop_sizes, resize_size, mean_sr, std_sr, mean_hr, std_hr):
        """
        crop_sizes: list[int]，随机裁剪尺寸列表，比如[800,900,1080]
        resize_size: (int,int)，裁剪后resize的固定尺寸，比如(1080,1080)
        mean_sr/std_sr, mean_hr/std_hr: 归一化参数
        """
        self.crop_sizes = crop_sizes
        self.resize_size = resize_size
        self.mean_sr = mean_sr
        self.std_sr = std_sr
        self.mean_hr = mean_hr
        self.std_hr = std_hr

    def __call__(self, img_sr, img_hr):
        # 随机选择裁剪尺寸
        crop_size = random.choice(self.crop_sizes)
        #print(f"Using crop size: {crop_size}")

        # 获取随机裁剪参数，保证SR和HR同步裁剪同一位置
        i, j, h, w = transforms.RandomCrop.get_params(img_sr, output_size=(crop_size, crop_size))

        img_sr = TF.crop(img_sr, i, j, h, w)
        img_hr = TF.crop(img_hr, i, j, h, w)

        # 裁剪后resize为固定大小
        img_sr = img_sr.resize(self.resize_size, resample=Image.BICUBIC)
        img_hr = img_hr.resize(self.resize_size, resample=Image.BICUBIC)

        # 随机水平翻转
        if random.random() > 0.5:
            img_sr = TF.hflip(img_sr)
            img_hr = TF.hflip(img_hr)

        # 转Tensor + Normalize
        img_sr = TF.to_tensor(img_sr)
        img_sr = TF.normalize(img_sr, self.mean_sr, self.std_sr)

        img_hr = TF.to_tensor(img_hr)
        img_hr = TF.normalize(img_hr, self.mean_hr, self.std_hr)

        return img_sr, img_hr


class MyDataSet(Dataset):
    """自定义数据集，返回：SR图像、HR图像、标签"""
    def __init__(self, sr_paths: list, hr_paths: list, labels: list,
                 paired_transform=None, transform_val_sr=None, transform_val_hr=None, is_train=True):
        self.sr_paths = sr_paths
        self.hr_paths = hr_paths
        self.labels = labels
        self.paired_transform = paired_transform
        self.transform_val_sr = transform_val_sr
        self.transform_val_hr = transform_val_hr
        self.is_train = is_train

    def __len__(self):
        return len(self.sr_paths)

    def __getitem__(self, idx):
        sr_img = Image.open(self.sr_paths[idx]).convert('RGB')
        hr_img = Image.open(self.hr_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.is_train and self.paired_transform:
            sr_img, hr_img = self.paired_transform(sr_img, hr_img)
        else:
            sr_img = self.transform_val_sr(sr_img)
            hr_img = self.transform_val_hr(hr_img)

        return sr_img, hr_img, label


def parse_txt(txt_file, root_dir="C:\D\ImageSR\data\RealSRQ-KLTSRQA-released\RealSRQ"):
    sr_paths, hr_paths, labels = [], [], []
    for line in open(txt_file, 'r'):
        line = line.strip()
        if not line or '#' not in line:
            continue
        #sr_name, hr_name, score_str = line.split('#')
        sr_name,score_str,_, hr_name = line.split('#')
        label = float(score_str)

        sr_path = os.path.join(root_dir, "SR_results",sr_name) if root_dir else sr_name
        hr_path = os.path.join(root_dir, "HR", hr_name) if root_dir else hr_name

        sr_paths.append(sr_path)
        hr_paths.append(hr_path)
        labels.append(label)

    return sr_paths, hr_paths, labels

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
    mean1 = torch.zeros(6)
    std1 = torch.zeros(6)
    mean2 = torch.zeros(6)
    std2 = torch.zeros(6)
    i=0
    for X1,X2, _ in train_loader:
        print(i)
        i+=1
        for d in range(3):
            mean1[d] += X1[:, d, :, :].mean()
            std1[d] += X1[:, d, :, :].std()
            mean2[d] += X2[:, d, :, :].mean()
            std2[d] += X2[:, d, :, :].std()
    mean1.div_(len(train_data))
    std1.div_(len(train_data))
    mean2.div_(len(train_data))
    std2.div_(len(train_data))
    return list(mean1.numpy()), list(std1.numpy()), list(mean2.numpy()), list(std2.numpy())

# 均值和方差
mean_sr = [0.45175493, 0.45306018, 0.37391272]
std_sr  = [0.09589117, 0.08915779, 0.08248971]
mean_hr = [0.45255446, 0.45361045, 0.37407586]
std_hr  = [0.11673171, 0.109770544, 0.10265977]

# 多尺度裁剪 + resize + flip + normalize
paired_train_transform = PairedMultiScaleCropFlip(
    crop_sizes=[768, 900, 1080],
    resize_size=(768, 768),
    mean_sr=mean_sr,
    std_sr=std_sr,
    mean_hr=mean_hr,
    std_hr=std_hr
)

# 验证集和测试集采用固定resize + normalize
transform_sr = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.05605867, 0.050786886, 0.04435081], [0.028125687, 0.027141795, 0.02876606])
])
transform_hr = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.05608858, 0.050779864, 0.044556625], [0.027764285, 0.026925579, 0.028313829])
])

# 加载路径和标签
train_sr, train_hr, train_label = parse_txt('C:\D\ImageSR\data\RealSRQ-KLTSRQA-released\RealSRQ/train.txt')
val_sr, val_hr, val_label = parse_txt('C:\D\ImageSR\data\RealSRQ-KLTSRQA-released\RealSRQ/val.txt')
test_sr, test_hr, test_label = parse_txt('C:\D\ImageSR\data\RealSRQ-KLTSRQA-released\RealSRQ/test.txt')

# 构建 Dataset
#train_dataset = MyDataSet(train_sr, train_hr, train_label,
#                         paired_transform=paired_train_transform,
#                          is_train=True)
train_dataset = MyDataSet(train_sr, train_hr, train_label,
                        transform_val_sr=transform_sr,
                        transform_val_hr=transform_hr,
                        is_train=False)
val_dataset = MyDataSet(val_sr, val_hr, val_label,
                        transform_val_sr=transform_sr,
                        transform_val_hr=transform_hr,
                        is_train=False)
test_dataset = MyDataSet(test_sr, test_hr, test_label,
                        transform_val_sr=transform_sr,
                        transform_val_hr=transform_hr,
                        is_train=False)

# DataLoader
train_loader = data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

if __name__ == "__main__":
    print(getStat(train_dataset))