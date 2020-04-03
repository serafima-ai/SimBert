from simbert.losses.loss import Loss
import torch.nn as nn


class HingeEmbeddingLoss(Loss):

    def criterion(self, margin=1.0, size_average=None, reduce=None, reduction='mean') -> nn.HingeEmbeddingLoss:
        return nn.HingeEmbeddingLoss(margin=margin, size_average=size_average, reduce=reduce, reduction=reduction)
