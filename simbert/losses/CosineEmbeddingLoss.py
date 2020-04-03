from simbert.losses.loss import Loss
import torch.nn as nn


class CosineEmbeddingLoss(Loss):

    def criterion(self, margin=0.0, size_average=None, reduce=None, reduction='mean') -> nn.CosineEmbeddingLoss:
        return nn.CosineEmbeddingLoss(margin=margin, size_average=size_average, reduce=reduce, reduction=reduction)
