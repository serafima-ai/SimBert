import torch
from simbert.optimizers.optimizer import Optimizer


class Adam(Optimizer):

    def __init__(self, configs, ):
        Optimizer.__init__(self, configs)

    def optimizer(self, params, lr='2e-05', epsilon='1e-08') -> torch.optim.Adam:
        lr = self.configs.get('learning_rate', lr)
        epsilon = self.configs.get('epsilon', epsilon)

        return torch.optim.Adam(params, lr=lr, eps=epsilon)
