from simbert.losses.loss import Loss
import torch.nn as nn


class CTCLoss(Loss):

    def criterion(self, blank=0, reduction='mean', zero_infinity=False) -> nn.CTCLoss:
        return nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
