# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
import settings_Rain200L_real as settings
import glob
import random

class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.real_dir = os.path.join(settings.data_dir, 'Real')
        self.mat_files = os.listdir(self.root_dir)
        self.mat_real = os.listdir(self.real_dir)
        self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files) # 1800
        self.real_num = len(self.mat_real) # 466

    def __len__(self):
        # return self.file_num * 100
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        b, g, r = cv2.split(img_pair)
        img_pair = cv2.merge([r, g, b])

        # TODO: Real
        rad_num = random.randint(0, self.real_num - 1)
        real_name = self.mat_real[rad_num]
        real_img = cv2.imread(os.path.join(self.real_dir, real_name)).astype(np.float32) / 255


        if settings.aug_data:
            O, B, R = self.crop(img_pair, real_img, aug=True)
            O, B, R = self.flip(O, B, R)
            O, B, R = self.rotate(O, B, R)
        else:
            O, B, R = self.crop(img_pair, real_img, aug=False)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        R = np.transpose(R, (2, 0, 1))

        sample = {'O': O, 'B': B, 'R': R}

        return sample

    def crop(self, img_pair, real_img, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = ww // 2

        rh, rw, rc = real_img.shape

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)

        real_r = self.rand_state.randint(0, rh - p_h)
        real_c = self.rand_state.randint(0, rw - p_w)

        O = img_pair[r: r+p_h, c+w: c+p_w+w]
        B = img_pair[r: r+p_h, c: c+p_w]
        R = real_img[real_r: real_r+p_h, real_c: real_c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))
            R = cv2.resize(R, (patch_size, patch_size))

        return O, B, R

    def flip(self, O, B, R):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
            R = np.flip(R, axis=1)

        return O, B, R

    def rotate(self, O, B, R):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        R = cv2.warpAffine(R, M, (patch_size, patch_size))

        return O, B, R

class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        h, ww, c = img_pair.shape
        w = ww // 2
        O = np.transpose(img_pair[:, w:], (2, 0, 1))
        B = np.transpose(img_pair[:, :w], (2, 0, 1))
        sample = {'O': O, 'B': B}

        return sample

class ShowDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.img_files = sorted(os.listdir(self.root_dir))
        self.file_num = len(self.img_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255

        h, ww, c = img_pair.shape
        w = ww // 2

        if settings.pic_is_pair:
            O = np.transpose(img_pair[:, w:], (2, 0, 1))
            B = np.transpose(img_pair[:, :w], (2, 0, 1))
        else:
            O = np.transpose(img_pair[:, :], (2, 0, 1))
            B = np.transpose(img_pair[:, :], (2, 0, 1))

        sample = {'O': O, 'B': B,'file_name':file_name[:-4]}

        return sample


if __name__ == '__main__':
    dt = TrainValDataset('train')
    print('TrainValDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    dt = TestDataset('test')
    print('TestDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    print('ShowDataset')
    dt = ShowDataset('test')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())
