from dotmap import DotMap
from abc import ABCMeta, abstractmethod

from simbert.kernel import Kernel
from simbert.models import *


class Model(Kernel, metaclass=ABCMeta):
    configs = None

    train_dataset = None

    val_dataset = None

    test_dataset = None

    DataProcessor = None

    tokenizer = None

    model = None

    def __init__(self, configs: DotMap = None):
        if configs is not None:
            configs['models_path'] = configs.get("models_path", "/simbert_models/")
        self.configs = configs

    @abstractmethod
    def predict(self, inputs):
        """"""

    @abstractmethod
    def evaluate_model(self):
        """"""

    @abstractmethod
    def train_model(self):
        """"""

    def load_dataset(self):
        self.train_dataset, self.val_dataset, self.test_dataset = self.DataProcessor.prepare_dataset(self.tokenizer)

    @abstractmethod
    def new_tokenizer(self):
        """"""

    @abstractmethod
    def data_processor(self):
        """"""

    def apply_configs(self, configs: DotMap):
        self.configs = configs
        self.tokenizer = self.new_tokenizer()
        self.DataProcessor = self.data_processor()
