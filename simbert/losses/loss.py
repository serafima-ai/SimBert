from dotmap import DotMap
from abc import abstractmethod
from simbert.kernel import Kernel


class Loss(Kernel):
    configs = None

    def __init__(self, configs: DotMap = DotMap()):
        self.configs = configs

    @abstractmethod
    def criterion(self, size_average=None, reduce=None, reduction='mean'):
        """"""
