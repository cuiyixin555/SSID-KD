import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
import cv2
import numpy as np
import argparse
import glob
import os

parser = argparse.ArgumentParser(description='Image Merge')
parser.add_argument('--rain_path', type=str, default='/media/ubuntu/Seagate/RainData/Rain1200/test/small/rain/')
parser.add_argument('--norain_path', type=str, default='/media/ubuntu/Seagate/RainData/Rain1200/test/small/norain/')
parser.add_argument('--save_path', type=str, default='/media/ubuntu/Seagate/RainData/Rain1200_merge/test/')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default='0', help='GPU id')
opt = opt = parser.parse_args()

for i in range(1, 1201):
    rain_name = 'norain-%d.png' % (i)
    norain_name = 'norain-%d.png' % (i)

    rain_image = cv2.imread(os.path.join(opt.rain_path, rain_name))
    rh, rw, rc = rain_image.shape[0], rain_image.shape[1], rain_image.shape[2]

    norain_image = cv2.imread(os.path.join(opt.norain_path, norain_name))
    gh, gw, gc = norain_image.shape[0], norain_image.shape[1], norain_image.shape[2]

    if (rh == gh) and (rw == gw) and (rc == gc):
        merge_matrix = np.zeros((rh, 2*rw, rc))
        merge_matrix[:, :rw, :] = norain_image
        merge_matrix[:, rw:2*rw, :] = rain_image

    newname = 'norain-%d.png' % (i)
    savepath = os.path.join(opt.save_path, newname)

    cv2.imwrite(savepath, merge_matrix)