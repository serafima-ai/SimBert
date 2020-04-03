from simbert.losses.loss import Loss
import torch.nn as nn


class SmoothL1Loss(Loss):

    def criterion(self, size_average=None, reduce=None, reduction='mean') -> nn.SmoothL1Loss:
        return nn.SmoothL1Loss(size_average=size_average, reduce=reduce, reduction=reduction)
