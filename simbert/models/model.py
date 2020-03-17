from dotmap import DotMap

from simbert.kernel import Kernel
from simbert.models import *


class Model(Kernel):
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

    def evaluate(self):
        pass

    def train(self):
        pass

    def load_dataset(self):
        self.DataProcessor.prepare_dataset(self.tokenizer)
