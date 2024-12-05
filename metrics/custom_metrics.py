from math import log10

import torch.nn.functional as F
from piq import ssim


def compute_psnr(predicted, target):
    mse = F.mse_loss(predicted, target)
    psnr = 10 * log10(1 / mse.item())
    return psnr


def compute_ssim(predicted, target):
    predicted = (predicted + 1) / 2  # Normalize to [0, 1]
    target = (target + 1) / 2
    return ssim(predicted, target, data_range=1.0).item()

