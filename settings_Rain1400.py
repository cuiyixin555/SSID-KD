import os
import logging

ssim_loss = True
aug_data = False # Set as False for fair comparison

patch_size = 120
pic_is_pair = True  # input picture is pair or single

lr = 5e-4

data_dir = '/media/ubuntu/Seagate/RainData/Rain1400_merge/'
if pic_is_pair is False:
    data_dir = '/data1/wangcong/dataset/real-world-images'
log_dir = '../logdir_Rain1400'
show_dir = '../showdir'
model_dir = './trained_model/Rain1400/'
show_dir_feature = '../showdir_feature'

log_level = 'info'
model_path = os.path.join(model_dir, 'net_latest')
save_steps = 400

num_workers = 8

num_GPU = 2
device_id = '0,1'

epoch = 800 # 10
batch_size = 12

if pic_is_pair:
    root_dir = os.path.join(data_dir, 'train')
    mat_files = os.listdir(root_dir)
    num_datasets = len(mat_files)
    # l1 = 3150
    l1 = int(3/5 * epoch * num_datasets / batch_size) # 90000  # 6300
    # l1 = 76000 # Rain200H
    # l1 = 4200 # Rain1400
    # l1 = 3600
    l2 = int(4/5 * epoch * num_datasets / batch_size) # 120000 # 8400
    # l2 = 120000 # Rain200Hs
    # l2 = 0
    one_epoch = int(num_datasets/batch_size)
    total_step = int((epoch * num_datasets)/batch_size)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


