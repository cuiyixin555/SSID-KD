import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import cv2
import argparse
import numpy as np
import math
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
import settings_Rain200L_real as settings
from dataset_Rain200L_real import TrainValDataset, TestDataset
from New_model2 import ODE_DerainNet
from cal_ssim import SSIM

os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id
logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
# torch.cuda.set_device(settings.device_id)

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def PSNR(img1, img2):
    b, _, _, _ = img1.shape
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)  # +mse
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def save_image(derain, idx):
    derain_save = derain
    derain_save = torch.clamp(derain_save, 0.0, 1.0)
    derain_save_np = derain_save.cpu().numpy()
    derain_save_np = derain_save_np.squeeze()
    derain_save_np = derain_save_np.transpose(1, 2, 0)
    derain_save_np = np.uint8(derain_save_np * 255.0)
    cv2.imwrite(os.path.join(os.path.join('/media/ubuntu/Seagate/ACM_MM/experiment_res/Distillation/Rain200L+Real/Syn/', 'norain-%d.png'%(idx + 1))), derain_save_np)

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)
        if len(settings.device_id) > 1:
            self.net = nn.DataParallel(ODE_DerainNet()).cuda()
        else:
            torch.cuda.set_device(settings.device_id[0])
            self.net = ODE_DerainNet().cuda()

        self.l2 = MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.ssim = SSIM().cuda()
        self.dataloaders = {}

    def get_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        return self.dataloaders[dataset_name]

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])

    def inf_batch(self, name, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)

        with torch.no_grad():
            derain, fake_S, code = self.net(O)

        l1_loss = self.l1(derain, B)
        ssim = self.ssim(derain, B)
        psnr = PSNR(derain.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        losses = {'L1 loss': l1_loss}
        ssimes = {'ssim': ssim}
        return ssim, psnr, derain

def run_test(ckp_name):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name)
    dt = sess.get_dataloader('test')

    psnr_all = 0
    ssim_all = 0
    all_num = 0
    all_losses = {}

    for i, batch in enumerate(dt):
        ssim, psnr, derain = sess.inf_batch('test', batch)
        save_image(derain, i)
        psnr_all = psnr_all + psnr
        ssim_all = ssim_all + ssim
        batch_size = batch['O'].size(0)
        all_num += batch_size

    print('psnr_ll:%8f' % (psnr_all / all_num))
    print('ssim_ll:%8f' % (ssim_all / all_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='net_latest')

    args = parser.parse_args(sys.argv[1:])
    run_test(args.model)
