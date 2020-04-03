from simbert.losses.loss import Loss
import torch.nn as nn


class L1Loss(Loss):

    def criterion(self, size_average=None, reduce=None, reduction='mean') -> nn.L1Loss:
        return nn.L1Loss(size_average=size_average, reduce=reduce, reduction=reduction)
