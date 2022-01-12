# 一些通用的方法

import torch
import numpy as np


def sMAPE(x: torch.tensor, x_gt: torch.tensor) -> float:
    n = x.shape[0]
    channels = x.shape[1]
    assert n == x_gt.shape[0]
    x = x.numpy()
    x_gt = x_gt.numpy()
    # 转换成为numpy进行计算
    result = np.abs(x - x_gt) / (np.abs(x + x_gt) / 2)
    result = np.sum(result) / n
    result = result / channels
    return result * 100
