from dotmap import DotMap
from abc import abstractmethod
from simbert.kernel import Kernel


class Trainer(Kernel):
    configs = None

    def __init__(self, configs: DotMap = DotMap()):
        self.configs = configs

    @abstractmethod
    def trainer(self, max_epochs=1, gpus=-1, logger=None):
        """"""
