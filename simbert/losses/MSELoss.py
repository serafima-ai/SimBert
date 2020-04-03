from simbert.losses.loss import Loss
import torch.nn as nn


class MSELoss(Loss):

    def criterion(self, size_average=None, reduce=None, reduction='mean') -> nn.MSELoss:
        return nn.MSELoss(size_average=size_average, reduce=reduce, reduction=reduction)
