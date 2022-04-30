import os
import logging

patch_size = 120
pic_is_pair = True  # input picture is pair or single

lr = 5e-4

# data_dir = '/media/ubuntu/Seagate/RainDataSet/JORDER/RainTrainL/'
data_dir = '/media/ubuntu/Seagate/RainData/Rain200L_merge/'
log_dir = './logdir_Rain200L/'
model_dir = './trained_model/Rain200L/'
ssim_loss = True
aug_data = False

log_level = 'info'
model_path = os.path.join(model_dir, 'net_latest')
save_steps = 400

num_workers = 8

num_GPU = 2
device_id = '0,1'

epoch = 520 # 10
batch_size = 12

if pic_is_pair:
    root_dir = os.path.join(data_dir, 'train')
    mat_files = os.listdir(root_dir)
    num_datasets = len(mat_files)
    l1 = int(3/5 * epoch * num_datasets / batch_size)
    l2 = int(4 / 5 * epoch * num_datasets / batch_size)
    one_epoch = int(num_datasets/batch_size)
    total_step = int((epoch * num_datasets)/batch_size)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


