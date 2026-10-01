'''所有依赖与全局配置。与论文实验使用的环境一致。

论文：GRLV: Gradient Reversal Layer with Variance Loss for Major Depressive
Disorder Detection。
'''

import copy, os, random, json, time
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# 本目录下的 optuna.py 是本项目的搜参脚本，与官方 optuna 包同名。
# 导入时临时把本目录从 sys.path 中移除，保证拿到的是官方包。
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_saved_path = list(_sys.path)
_sys.path = [p for p in _sys.path if os.path.abspath(p or '.') != _HERE]
import optuna
_sys.path = _saved_path
import torchmetrics
from munch import Munch, munchify
from einops import rearrange, repeat

import braindecode
import braindecode.models as bm


# ----------------------------------------------------------------------
# 路径（可用环境变量覆盖，不写死任何机器相关的位置）
#   GRLV_DATA_FILE : dataset_and_info_256.pt 的路径
#   GRLV_OUT_DIR   : 结果输出目录
# ----------------------------------------------------------------------
DATA_FILE = os.environ.get('GRLV_DATA_FILE', 'data/dataset_and_info_256.pt')
OUT_DIR = os.environ.get('GRLV_OUT_DIR', 'output')
DOMAIN_FILE = os.path.join(OUT_DIR, 'domains.npz')          # 固化的域标签

# ----------------------------------------------------------------------
# 实验设定（与论文 4.3、4.4 一致）
# ----------------------------------------------------------------------
N_MDD, N_HC = 45, 45            # 论文使用 45 名 MDD + 45 名 HC
N_SENSORS, N_TIMES = 275, 256   # 单个试次 -0.525 s ~ 0.75 s @ 200 Hz
N_FACES = 3                     # 中性、悲伤、高兴，按此顺序拼接
N_CHANS = N_SENSORS * N_FACES   # 825
SFREQ = 200.0
N_SUBJS = 153                   # 个体分类头输出的被试数
K_DOMAINS = 3                   # 论文使用的域数量

N_OUTER_FOLDS = 6               # 嵌套交叉验证外层折数
N_INNER_VALID = 4               # 内层验证组数量
N_CONCAT = 50                   # 每个被试抽取的拼接样本数

SEED = 42
BATCH_SIZE = 96
MAX_EPOCHS = 100

# 超参搜索空间（回复 Comment 18）
SEARCH_SPACE = {
    'label_opt_lr':  (1e-8, 1e-1, True),
    'grl_gamma':     (1e-3, 3.0, True),    # 论文正文写作 [10^-3, 3]
    'std_gamma':     (1e-3, 3.0, True),
    'max_grl_lambd': (1e-3, 1e2, True),
    'max_std_lambd': (1e-3, 1e2, True),
}
N_TRIALS = 50                   # 每个验证组的 trial 数
N_STARTUP_TRIALS = 5            # 前 5 个 trial 随机采样，其余由 TPE 给出
TOP_N_AVERAGE = 2               # 每个外层折取前 2 组超参做平均

# 剪枝规则（回复 Comment 18-5）
PRUNE_AFTER = 10                # 从第 10 个 iteration 开始
PRUNE_WINDOW = 5                # 用最近 5 个 iteration 的验证准确率拟合直线
PRUNE_TRAIN_ACC = 0.99          # 训练准确率到 0.99 也剪枝

MODELS = ['AttentionBaseNet', 'EEGMiner', 'EEGNet', 'BrainModule',
          'EEGConformer', 'BIOT', 'FBCNet', 'IFNet', 'MSVTNet', 'SSTDPN']

# ablation 取值：full / no_subj_grl_var / no_domain_grl_bce / baseline
ABLATIONS = ['full', 'no_subj_grl_var', 'no_domain_grl_bce', 'baseline']

# ----------------------------------------------------------------------
# t-SNE 设定（回复 Comment 20）
# ----------------------------------------------------------------------
TSNE_SETTINGS = dict(n_components=2, perplexity=30, init='pca',
                     metric='euclidean', learning_rate='auto',
                     early_exaggeration=12, max_iter=1000, random_state=SEED)
DOMAIN_COLORS = ['#d62728', '#2ca02c', '#1f77b4']   # 红、绿、蓝


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g
