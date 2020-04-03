from simbert.losses.loss import Loss
import torch.nn as nn


class PoissonNLLLoss(Loss):

    def criterion(self, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None,
                  reduction='mean') -> nn.PoissonNLLLoss:
        return nn.PoissonNLLLoss(log_input=log_input, full=full, size_average=size_average, eps=eps,
                                 reduce=reduce, reduction=reduction)
