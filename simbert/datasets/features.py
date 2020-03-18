from dotmap import DotMap

from simbert.kernel import Kernel


class FeaturesProcessor(Kernel):
    configs = None

    def __init__(self, configs: DotMap = DotMap()):
        self.configs = configs

    def convert_examples_to_features(self, data, tokenizer):
        pass
