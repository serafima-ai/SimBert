from simbert.kernel import Kernel


class FeaturesProcessor(Kernel):
    configs = None

    def convert_examples_to_features(self, data, tokenizer):
        pass
