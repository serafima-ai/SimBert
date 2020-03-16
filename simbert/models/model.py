from simbert.kernel import Kernel
from simbert.models import *


class Model(Kernel):

    configs = {}

    train_dataset = None

    val_dataset = None

    test_dataset = None

    DataProcessor = None

    tokenizer = None

    model = None

    def __init__(self, configs: dict):
        configs['models_path'] = configs.get("models_path", "/simbert_models/")

    def evaluate(self):
        pass

    def train(self):
        pass

    def load_dataset(self):
        self.DataProcessor.prepare_dataset(self.tokenizer)
