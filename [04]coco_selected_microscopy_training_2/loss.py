from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gc
import os
from datetime import datetime


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.contiguous().view(pred.shape[0], -1)
        target_flat = target.contiguous().view(target.shape[0], -1)
        intersection = (pred_flat * target_flat).sum(1)
        denominator = pred_flat.pow(2).sum(1) + target_flat.pow(2).sum(1)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class WeightedDiceBCELoss(nn.Module):
    def __init__(self, dice_weight=2.0, bce_weight=1.0):
        super(WeightedDiceBCELoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        return self.dice_weight * self.dice(pred, target) + self.bce_weight * self.bce(pred, target)


class MultiTaskLoss(nn.Module):
    def __init__(self, dice_weight=2.0, bce_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.seg_loss = WeightedDiceBCELoss(dice_weight, bce_weight)
        self.heatmap_loss = nn.MSELoss()
        #self.count_loss = nn.MSELoss()

    def forward(self, seg_pred, seg_target, heatmap_pred, heatmap_target): #, count_pred, count_target
        l_seg = self.seg_loss(seg_pred, seg_target)
        l_heat = self.heatmap_loss(heatmap_pred, heatmap_target)
        #l_count = self.count_loss(count_pred.view(-1), count_target.view(-1).float())
        print(l_seg, l_heat)#,l_count)
        return l_seg + 2*l_heat , (l_seg.item(), l_heat.item())#, l_count.item())
