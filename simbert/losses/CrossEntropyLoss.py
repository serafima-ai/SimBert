from simbert.losses.loss import Loss
import torch.nn as nn


class CrossEntropyLoss(Loss):

    def criterion(self, weight=None, size_average=None, ignore_index=-100, reduce=None,
                  reduction='mean') -> nn.CrossEntropyLoss:
        return nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index,
                                   reduce=reduce, reduction=reduction)
