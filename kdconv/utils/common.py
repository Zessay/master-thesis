# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-04
import os
import random
import numpy as np
import torch
from typing import Dict, List


def seed_everything(seed: int = 2020):
    """设置整个开发环境的seed"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def save_losses(basename: str, losses: Dict[str, List[float]]):
    for key, value in losses.items():
        with open(os.path.join(basename, f"{key}.npy"), "wb") as f:
            np.save(f, value)
