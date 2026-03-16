import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import torchvision.transforms.functional as TF

class DehazeDataset(Dataset):
    def __init__(self, root_dir, mode='train', crop_size=512):
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size

        self.hazy_dir = os.path.join(root_dir, 'hazy')
        # 兼容 gt 或 clear 文件夹命名
        if os.path.exists(os.path.join(root_dir, 'gt')):
            self.clear_dir = os.path.join(root_dir, 'gt')
        else:
            self.clear_dir = os.path.join(root_dir, 'clear')

        self.hazy_files = self._get_image_files(self.hazy_dir)
        self.clear_files = self._get_image_files(self.clear_dir)
        self.hazy_files.sort()
        self.clear_files.sort()

        assert len(self.hazy_files) == len(self.clear_files), \
            f"数量不匹配: hazy({len(self.hazy_files)}) != clear({len(self.clear_files)})"

    def _get_image_files(self, dir_path):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        return [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in valid_extensions]

    def _augment(self, hazy, clear):
        # 随机裁剪
        w, h = hazy.size
        if w > self.crop_size and h > self.crop_size:
            i, j, h_crop, w_crop = tfs.RandomCrop.get_params(
                hazy, output_size=(self.crop_size, self.crop_size))
            hazy = TF.crop(hazy, i, j, h_crop, w_crop)
            clear = TF.crop(clear, i, j, h_crop, w_crop)
        else:
            hazy = TF.resize(hazy, (self.crop_size, self.crop_size))
            clear = TF.resize(clear, (self.crop_size, self.crop_size))

        # 随机翻转
        if random.random() > 0.5:
            hazy = TF.hflip(hazy)
            clear = TF.hflip(clear)

        if random.random() > 0.5:
            hazy = TF.vflip(hazy)
            clear = TF.vflip(clear)

        # 随机旋转
        rotations = [0, 90, 180, 270]
        angle = random.choice(rotations)
        if angle > 0:
            hazy = TF.rotate(hazy, angle)
            clear = TF.rotate(clear, angle)

        return hazy, clear

    def _add_night_noise(self, tensor):
        # 仅在训练时添加极微量的噪声，防止过拟合，但不破坏暗部细节
        if random.random() > 0.5:
            # 【关键修改】大幅降低噪声水平 (0.05 -> 0.005)
            noise_level = random.uniform(0, 0.005)
            noise = torch.randn_like(tensor) * noise_level
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0, 1)
        return tensor

    def __getitem__(self, index):
        hazy_name = self.hazy_files[index]
        clear_name = self.clear_files[index]

        hazy_img = Image.open(os.path.join(self.hazy_dir, hazy_name)).convert('RGB')
        clear_img = Image.open(os.path.join(self.clear_dir, clear_name)).convert('RGB')

        if self.mode == 'train':
            hazy_img, clear_img = self._augment(hazy_img, clear_img)
            hazy_tensor = TF.to_tensor(hazy_img)
            clear_tensor = TF.to_tensor(clear_img)
            # 添加微量噪声
            hazy_tensor = self._add_night_noise(hazy_tensor)
        else:
            hazy_img = TF.resize(hazy_img, (self.crop_size, self.crop_size))
            clear_img = TF.resize(clear_img, (self.crop_size, self.crop_size))
            hazy_tensor = TF.to_tensor(hazy_img)
            clear_tensor = TF.to_tensor(clear_img)

        return hazy_tensor, clear_tensor

    def __len__(self):
        return len(self.hazy_files)