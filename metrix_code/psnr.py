import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import math
import argparse
import glob
import os

parser = argparse.ArgumentParser(description="ceshi_psnr")
parser.add_argument("--res_path", type=str, default='')
parser.add_argument("--gt_path", type=str, default='')
opt = parser.parse_args()

def PSNR(img1, img2):
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)

    if mse == 0:
        return 100

    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == '__main__':
    psnr_sum = 0.0

    for i in range(1, 201):
        gt_name = 'norain-%d.png' % (i)
        res_name = 'norain-%d.png' % (i)

        gt_image = cv2.imread(os.path.join(opt.gt_path, gt_name))
        res_image = cv2.imread(os.path.join(opt.res_path, res_name))

        psnr = PSNR(res_image, gt_image)
        print('psnr: ', psnr)

        psnr_sum = psnr_sum + psnr

    psnr_avg = psnr_sum / 200.0
    print('psnr_avg: ', psnr_avg)


