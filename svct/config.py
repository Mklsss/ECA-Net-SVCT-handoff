import torch
import numpy as np

#数据
DATA_DIR = '/root/autodl-fs/dataset/'
TRAIN_FILE = 'train_meiaonew.npz'
TEST_FILE = 'test_meiaonew.npz'




OUTPUT_DIR = 'checkpoints/' 

#CT/正弦图参数
IMAGE_SIZE = 256
NUM_ANGLES_FULL = 360 
NUM_DETECTORS = 357  
num_sparse_views = 120 ###########

num_full_views = 360 
SPARSE_VIEW_ANGLES = np.linspace(0, num_full_views, num_sparse_views, endpoint=False, dtype=int)
SPARSE_VIEW_INDICES = np.linspace(0, NUM_ANGLES_FULL - 1, num_sparse_views, endpoint=False, dtype=int)

TRAIN_FBP_FILE = f"train_fbp_{len(SPARSE_VIEW_ANGLES)}views.npz"
TEST_FBP_FILE  = f"test_fbp_{len(SPARSE_VIEW_ANGLES)}views.npz"

IMG_CHANNELS = 1

# FBP参数 
ANGLES_RAD = np.linspace(0., 2 * np.pi, NUM_ANGLES_FULL, endpoint=False) 

SPARSE_ANGLES_RAD = ANGLES_RAD[SPARSE_VIEW_INDICES]

# 训练超参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 24#原来是8 
LEARNING_RATE_CNN = 1e-4
LEARNING_RATE_G = 1e-4    #8e-5 #5e-5   
LEARNING_RATE_D = 1e-5    #7e-6 #5e-6   
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
NUM_WORKERS = 16#原来是8 
EPOCHS_CNN = 100
EPOCHS_GAN = 50
PRETRAIN_GAN_MSE_EPOCHS = 20

D_UPDATE_FREQ = 1 
GRAD_CLIP_VALUE = 1.0 
NUM_TEST_VISUALS = 5

LABEL_SMOOTHING_REAL = 0.8
LABEL_SMOOTHING_FAKE = 0.2 

# GAN损失函数权重 

LAMBDA_MSE = 0.8        # 生成器中像素级MSE/L1损失的权重
LAMBDA_ADV = 0.05       #  对抗损失的权重
LAMBDA_CONTENT = 0.0001  # VGG感知/内容损失的权重
LAMBDA_TV = 1e-7        # 全变分损失的权重 (用于平滑图像)

#数据归一化范围
normalize_img = (-1, 1)
normalize_sino = (0, 1) 

# VGG内容损失配置
VGG_FEATURE_LAYER = 35 

# FBP
GEOMETRY = 'fanflat'
DETECTORS_COUNT = 357
CENTER_OF_ROTATION = DETECTORS_COUNT / 2.0
DETECTOR_WIDTH = 1   
DSD = 605.0           
DSO = 635.0            

# 可视化
SAVE_INTERVAL = 20
VISUALIZE_INTERVAL = 5 

# Generator
GEN_FEATURES = 64              
GEN_NUM_BLOCKS = 4             
GEN_BOTTLENECK_EXPANSION = 4  
USE_ECA_IN_GENERATOR = False ####################False   
ECA_KERNEL_SIZE = 3          
IMG_CHANNELS_GEN = 1
# Discriminator 
DISC_FEATURES_START = 64  
IMG_CHANNELS_DISC = 1 
