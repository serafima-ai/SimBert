from simbert.losses.loss import Loss
import torch.nn as nn


class MultiMarginLoss(Loss):

    def criterion(self, p=1, margin=1.0, weight=None, size_average=None, reduce=None,
                  reduction='mean') -> nn.MultiMarginLoss:
        return nn.MultiMarginLoss(p=p, margin=margin, weight=weight, size_average=size_average, reduce=reduce,
                                  reduction=reduction)
