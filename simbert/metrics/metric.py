from dotmap import DotMap

from simbert.kernel import Kernel


class Metric(Kernel):
    configs = None

    avg_supported = [None, 'binary', 'micro', 'macro', 'samples', 'weighted', 'noAns', 'hasAns']

    def __init__(self, configs: DotMap = DotMap(), name: str = ''):
        self.configs = configs

        self.average = None

        average = self.average_from_name(name)

        self.name = name

        if average in self.avg_supported:
            self.average = average

    def evaluate(self, y_true, y_pred):
        """"""

    def average_from_name(self, name: str) -> str:
        return name.split('_')[-1]

    def get_supported_avg(self) -> list:
        return self.avg_supported

    def get_metric_name(self) -> str:
        return self.name

    @classmethod
    def get_class_name(cls, name) -> str:

        class_name = ''

        for word in name.split('_'):
            if word not in cls.avg_supported:
                class_name += word[0].upper() + word[1:]

        return class_name
