from simbert.kernel import Kernel
from simbert.models import *


class Model(Kernel):

    configs = {}

    train_set = []

    test_set = []

    def __init__(self, configs: dict):
        configs['models_path'] = configs.get("models_path", "/simbert_models/")

    def evaluate(self):
        pass

    def train(self):
        pass


