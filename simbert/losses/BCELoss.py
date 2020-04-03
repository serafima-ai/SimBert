from simbert.losses.loss import Loss
import torch.nn as nn


class BCELoss(Loss):

    def criterion(self, weight=None, size_average=None, reduce=None, reduction='mean') -> nn.BCELoss:
        return nn.BCELoss(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction)
