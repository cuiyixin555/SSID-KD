import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import math
import argparse
import glob
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

parser = argparse.ArgumentParser(description="ceshi_ssim")
parser.add_argument("--res_path", type=str, default='')
parser.add_argument("--gt_path", type=str, default='')
opt = parser.parse_args()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # print(img1.size())
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


if __name__ == '__main__':
    ssim_sum = 0.0

    for i in range(1, 201):
        gt_name = 'norain-%d.png' % (i)
        res_name = 'norain-%d.png' % (i)

        gt_image = cv2.imread(os.path.join(opt.gt_path, gt_name))
        gt_image = np.float32(gt_image / 255.0)
        gt_image = gt_image.transpose(2, 0, 1)
        gt_image = np.expand_dims(gt_image, 0)
        gt_image_tensor = torch.from_numpy(gt_image)

        res_image = cv2.imread(os.path.join(opt.res_path, res_name))
        res_image = np.float32(res_image / 255.0)
        res_image = res_image.transpose(2, 0, 1)
        res_image = np.expand_dims(res_image, 0)
        res_image_tensor = torch.from_numpy(res_image)

        ssim = SSIM()
        ssim_value = ssim(res_image_tensor, gt_image_tensor)
        print('ssim: ', ssim_value)

        ssim_sum = ssim_sum + ssim_value

    ssim_avg = ssim_sum / 200.0
    print('ssim_avg: ', ssim_avg)


