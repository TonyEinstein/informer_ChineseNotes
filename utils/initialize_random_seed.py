# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     :   陈汝海                               
# @software   : PyCharm      
# @file       :   initialize_random_seed.py
# @Time       :   2021/11/12 15:18


import torch
import numpy as np
import random
from torch.backends import cudnn


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    cudnn.deterministic = True
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False