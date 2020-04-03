from simbert.losses.loss import Loss
import torch.nn as nn


class TripletMarginLoss(Loss):

    def criterion(self, margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None,
                  reduction='mean') -> nn.TripletMarginLoss:
        return nn.TripletMarginLoss(p=p, margin=margin, eps=eps, swap=swap, size_average=size_average, reduce=reduce,
                                    reduction=reduction)
