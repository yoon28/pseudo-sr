import os, sys
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as nnF
import torch

import glob

class faces_data(Dataset):
    def __init__(self, data_lr, data_hr=None, rgb=True, shuffle=True, z_size=[8, 8]):
        if data_hr is not None:
            self.hr_files = glob.glob(os.path.join(data_hr, "**/*.jpg"), recursive=True)
            self.hr_files.sort()
        self.lr_files = glob.glob(os.path.join(data_lr, "**/*.jpg"), recursive=True)
        self.lr_files.sort()
        if shuffle:
            if data_hr is not None:
                np.random.shuffle(self.hr_files)
            np.random.shuffle(self.lr_files)
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.training = data_hr is not None
        self.rgb = rgb
        self.z_size = z_size

    def __len__(self):
        if self.training:
            return min([len(self.lr_files), len(self.hr_files)])
        else:
            return len(self.lr_files)

    def __getitem__(self, index):
        if self.training:
            hr_idx = index % len(self.hr_files)
            hr = cv2.imread(self.hr_files[hr_idx])
        lr_idx = index % len(self.lr_files)
        lr = cv2.imread(self.lr_files[lr_idx])
        if self.rgb:
            if self.training:
                hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        data = dict()
        if np.prod(self.z_size) > 0:
            data["z"] = torch.randn(1, *self.z_size, dtype=torch.float32)
        data["lr"] = self.preproc(lr)
        data["lr_upx2"] = nnF.interpolate(data["lr"].unsqueeze(0), scale_factor=2, mode="bicubic", align_corners=False).clamp(min=0, max=1.0).squeeze(0)
        if self.training:
            data["hr"] = self.preproc(hr)
            data["hr_down"] = nnF.interpolate(data["hr"].unsqueeze(0), scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=False).clamp(min=0, max=1.0).squeeze(0)
        return data

    def get_noises(self, n):
        return torch.randn(n, 1, 64, dtype=torch.float32)

    def permute_data(self):
        if self.training:
            np.random.shuffle(self.hr_files)
        np.random.shuffle(self.lr_files)


if __name__ == "__main__":
    high_folder = os.path.join(os.environ["DATA_TRAIN"], "HIGH")
    low_folder = os.path.join(os.environ["DATA_TRAIN"], "LOW")
    test_folder = os.path.join(os.environ["DATA_TEST"])
    data = faces_data(low_folder, high_folder)
    for i in range(len(data)):
        d = data[i]
        for elem in d:
            if elem == 'z': continue
            img = d[elem].numpy().transpose(1, 2, 0)
            cv2.imshow(elem, img[:,:,::-1])
        cv2.waitKey()
    print("fin.")
