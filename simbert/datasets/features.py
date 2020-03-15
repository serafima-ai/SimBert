from simbert.kernel import Kernel


class Processor(Kernel):

    def __init__(self, config: dict):
        config['models_path'] = config.get("models_path", "/simbert_models/")
