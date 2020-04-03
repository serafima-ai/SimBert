from simbert.losses.loss import Loss
import torch.nn as nn


class BCEWithLogitsLoss(Loss):

    def criterion(self, weight=None, size_average=None, reduce=None, reduction='mean',
                  pos_weight=None) -> nn.BCEWithLogitsLoss:
        return nn.BCEWithLogitsLoss(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction,
                                    pos_weight=pos_weight)
