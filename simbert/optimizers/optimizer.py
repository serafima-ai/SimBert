from dotmap import DotMap
from abc import abstractmethod
from simbert.kernel import Kernel


class Optimizer(Kernel):
    configs = None

    def __init__(self, configs: DotMap = DotMap()):
        self.configs = configs

    @abstractmethod
    def optimizer(self, params, lr='2e-05', epsilon='1e-08'):
        """"""
