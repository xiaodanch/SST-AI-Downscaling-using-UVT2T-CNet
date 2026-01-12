import torch
import torch.nn as nn
import torchvision
import numpy

class RLoss(nn.Module):
    """loss = 1-R, Pearson correlation"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # check NaN or inf
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise ValueError("pred contains NaN or inf.")
        if torch.isnan(target).any() or torch.isinf(target).any():
            raise ValueError("target contains NaN or inf.")
        # mean
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        # cov
        cov = torch.mean((pred-pred_mean)*(target-target_mean))
        # std
        pred_std = torch.std(pred)
        target_std = torch.std(target)
        # chack std
        if pred_std < self.eps or target_std < self.eps:
            raise ValueError('small std')

        corr = cov / (pred_std * target_std + self.eps)  # avoid devide by 0
        # check corr
        if torch.isnan(corr) or torch.isinf(corr):
            raise ValueError('corr is nan or inf')
        corr = (corr+1) / 2 #[-1,1]->[0,1]
        return 1-corr