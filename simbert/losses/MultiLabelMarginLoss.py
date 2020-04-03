from simbert.losses.loss import Loss
import torch.nn as nn


class MultiLabelMarginLoss(Loss):

    def criterion(self, size_average=None, reduce=None, reduction='mean') -> nn.MultiLabelMarginLoss:
        return nn.MultiLabelMarginLoss(size_average=size_average, reduce=reduce, reduction=reduction)
