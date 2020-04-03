from simbert.losses.loss import Loss
import torch.nn as nn


class KLDivLoss(Loss):

    def criterion(self, size_average=None, reduce=None, reduction='mean') -> nn.KLDivLoss:
        return nn.KLDivLoss(size_average=size_average, reduce=reduce, reduction=reduction)
