# SSID-KD

## [Semi-Supervised Image Deraining using Knowledge Distillation]

### Introduction
Image deraining has achieved great progress by adopting supervised learning based on synthetic training pairs, which are usually limited in handling real-world rainy images.
Recently, semi-supervised methods are suggested to exploit real-world rainy images when training deep deraining models, but their performances are still notably inferior.
In this paper, we propose to address this crucial issue for image deraining in terms of the strategy of semi-supervised learning and backbone architecture.
First, as for semi-supervised learning, we propose to enforce the consistency of feature distribution of rain streaks in synthetic and real-world rainy images by knowledge distillation, so as to achieve semi-supervised image deraining with knowledge distillation (SSID-KD).
Then, as for the backbone in SSID-KD, we propose a novel multi-scale fusion module and pyramid fusion module to better extract deep features of rainy images.
Extensive experiments on both synthetic and real-world rainy images have validated that our SSID-KD not only can achieve better deraining results than existing semi-supervised deraining methods but also outperform state-of-the-art supervised deraining methods.
Profiting from well exploiting real-world rainy images, our SSID-KD can obtain more visually plausible deraining results.


## Prerequisites
- Python 3.6, PyTorch >= 1.0.0 
- Requirements: opencv-python, 
- Platforms: Ubuntu 16.04, cuda-10.0 & cuDNN v-7.3 (higher versions also work well)


## Datasets

SID and SSID-KD are trained on five synthetic datasets*: 
Rain200H [1], Rain200L [1], Rain800[2], Rain1200 [3] and Rain1400 [4]. 
and adopt Real-World datasets* for trained based Knowledge Distillation:
SPADatasets[5], Real-275.

## Getting Started

### 1) Testing

We have placed our pre-trained models trained by Rain200H into `./trained_model/Rain200H/`.

We have placed our pretrained models trained by knowledge distillation into `./trained_model/Rain200H+Real/`.

Run shell scripts to test the models:
```bash
python eval_Rain200H        # test models on Rain200H in SID
python eval_Rain200H_real   # test models on Rain200H in SSID-KD
python eval_Rain200L        # test models on Rain200L in SID
python eval_Rain200L_real   # test models on Rain200L in SSID-KD 
python eval_Rain1200        # test models on Rain1200 in SID
python eval_Rain1200_real   # test models on Rain1200 in SSID-KD
python eval_Rain1400        # test models on Rain1400 in SID
python eval_Rain1400_real   # test models on Rain1400 in SSID-KD
``` 

### 2) Evaluation metrics

We also provide the metric code in project: psnr.py and ssim.py.

```Matlab
 cd ./metrix_code
 python psnr.py
 python ssim.py
```
 
###
Average PSNR/SSIM values on four datasets:

Dataset      |NLEDN[6]     |ReHEN[7]     |PReNet[8]    |RPDNet[9]    |MSPFN[10]    |Syn2Real[11] |SSID-KD(Ours)
-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------
Rain200H     |27.315/0.8904|27.525/0.8663|27.640/0.8872|27.909/0.8923|25.554/0.8039|14.495/0.4021|28.925/0.9079
Rain200L     |37.539/0.9826|34.266/0.9660|36.488/0.9792|38.348/0.9849|30.367/0.9219|31.035/0.9365|38.778/0.9864
Rain1200     |30.799/0.9127|30.456/0.8702|27.307/0.8712|26.486/0.8401|32.273/0.9111|28.812/0.8400|32.288/0.9293
Rain1400     |31.014/0.9206|30.984/0.9156|30.609/0.9181|30.772/0.9178|31.016/0.9164|28.582/0.8586|31.037/0.9212
SPA          |30.596/0.9363|32.652/0.9297|32.720/0.9317|32.803/0.9337|30.174/0.9201|31.824/0.9307|33.026/0.9364

Dataset      |NLEDN[6]     |ReHEN[7]     |PReNet[8]    |SIRR[12]     |RPDNet[9]    |MSPFN[10]    |Syn2Real[11] |SSID-KD(Ours)
-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------
SPA          |7.1806       |7.1281       |7.1949       |5.5571       |7.5047       |7.9280       |7.1190       |7.0006
Real-275     |3.5554       |3.7355       |3.7745       |3.5492       |3.8957       |3.8616       |4.0372       |3.4739

### 3) Training

Run shell scripts to train the models:
```bash
python train_Rain200H.py      # training Teacher Network 
python train_Rain200H_real.py # training Student Network 
```
You can use `tensorboard --logdir ./trained_model/your_model_path` to check the training procedures. 

### Model Configuration

The following tables provide the configurations of options. 

#### Training Mode Configurations

Option                 |Default        | Description
-----------------------|---------------|------------
batchSize              | 12            | Training batch size
epochs                 | 520           | Number of training epochs
milestone              | [312, 416]    | When to decay learning rate
lr                     | 5e-4          | Initial learning rate
use_GPU                | True          | use GPU or not
gpu_id                 | 0,1           | GPU id
data_path              | N/A           | path to training images
save_path              | N/A           | path to save models and status           

#### Testing Mode Configurations

Option                 |Default           | Description
-----------------------|------------------|------------
use_GPU                | True             | use GPU or not
gpu_id                 | 0,1                | GPU id
logdir                 | N/A              | path to trained model
data_path              | N/A              | path to testing images
save_path              | N/A              | path to save results

## References
[1] W Yang, RT Feng, J Liu, Z Guo, and S Yan. Deep joint rain detection and removal from a single image. In IEEE CVPR 2017.

[2] Zhang, He and Sindagi, Vishwanath and Patel, Vishal M. Image De-Raining Using a Conditional Generative Adversarial Network. In IEEE TCSVT 2020.

[3] H Zhang, VM Patel. Density-Aware Single Image De-raining Using a Multi-stream Dense Network. In IEEE CVPR 2018.

[4] X Fu, J Huang, D Zeng, Y Huang, X Ding and J Paisley. Removing rain from single images via a deep detail network. In CVPR 2017.

[5] T Wang, X Yang, K Xu, S Chen, Q Zhang, and RWH Lau. Spatial attentive single-image deraining with a high quality real rain dataset. In IEEE CVPR 2019.

[6] Li, Guanbin and He, Xiang and Zhang, Wei and Chang, Huiyou and Dong, Le and Lin, Liang. Non-Locally Enhanced Encoder-Decoder Network for Single Image De-Raining. In ACM MM 2018.

[7] Yang, Youzhao and Lu, Hong. Single Image Deraining via Recurrent Hierarchy Enhancement Network. In ACM MM 2019.

[8] Ren, Dongwei and Zuo, Wangmeng and Hu, Qinghua and Zhu, Pengfei and Meng, Deyu. Progressive Image Deraining Networks: A Better and Simpler Baseline. In IEEE CVPR 2019.

[9] Pang, Bo and Zhai, Deming and Jiang, Junjun and Liu, Xianming. Single Image Deraining via Scale-Space Invariant Attention Neural Network. In ACM MM 2020.

[10] K Jiang, Z Wang, P Yi, C Chen, B Huang, Y Luo, J Ma and J Jiang. Multi-Scale Progressive Fusion Network for Single Image Deraining. In IEEE CVPR 2020.

[11] R Yasarla, VA Sindagi and VM Patel. Syn2Real Transfer Learning for Image Deraining Using Gaussian Processes. In IEEE CVPR 2020.

[12] W Wei, D Meng, Q Zhao, Z Xu and Y Wu. Semi-Supervised Transfer Learning for Image Rain Removal. In IEEE CVPR 2019.


