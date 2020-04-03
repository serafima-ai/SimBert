from simbert.losses.loss import Loss
import torch.nn as nn


class SoftMarginLoss(Loss):

    def criterion(self, size_average=None, reduce=None, reduction='mean') -> nn.SoftMarginLoss:
        return nn.SoftMarginLoss(size_average=size_average, reduce=reduce, reduction=reduction)
