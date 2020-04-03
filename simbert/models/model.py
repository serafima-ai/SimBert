from dotmap import DotMap
from abc import ABCMeta, abstractmethod

from simbert.kernel import Kernel
from simbert.losses.loss import Loss
from simbert.metrics.metric import Metric


class Model(Kernel, metaclass=ABCMeta):

    metrics = {}

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

        self.loss_func = None

        self.set_metrics()

        self.set_loss_func()

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

    def set_metrics(self, metrics=None):
        if metrics is None:
            metrics = self.configs.get('metrics', {})

        M = Metric()

        for metric in metrics:
            metric_class = M.get(M.get_class_name(metric))

            if metric_class is not None:
                self.metrics.update({metric: metric_class(name=metric)})

    def set_loss_func(self, loss=None):
        if loss is None and type(self.configs.loss.loss_func_name) is not DotMap:
            loss = self.configs.loss.get('loss_func_name', 'CrossEntropyLoss')

        L = Loss()

        loss_class = L.get(loss)

        if loss_class is not None:
            self.loss_func = loss_class().criterion()

    def calculate_metrics(self, y_true, y_pred, stage='', apply=None) -> dict:
        validation_scores = {}

        if stage is not '':
            stage += '_'

        for _, metric in self.metrics.items():
            val = metric.evaluate(y_true, y_pred)
            if apply is not None:
                val = apply(val)
            validation_scores.update({stage + metric.get_metric_name(): val})

        return validation_scores
