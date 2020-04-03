from simbert.losses.loss import Loss
import torch.nn as nn


class MultiLabelSoftMarginLoss(Loss):

    def criterion(self, weight=None, size_average=None, reduce=None, reduction='mean') -> nn.MultiLabelSoftMarginLoss:
        return nn.MultiLabelSoftMarginLoss(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction)
