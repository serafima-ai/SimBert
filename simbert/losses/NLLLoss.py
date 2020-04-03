from simbert.losses.loss import Loss
import torch.nn as nn


class NLLLoss(Loss):

    def criterion(self, weight='', size_average=None, ignore_index=-100, reduce=None, reduction='mean') -> nn.NLLLoss:
        return nn.NLLLoss(weight=weight, size_average=size_average, ignore_index=ignore_index,
                          reduce=reduce, reduction=reduction)
