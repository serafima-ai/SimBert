import logging
from dotmap import DotMap
from abc import ABCMeta, abstractmethod

from simbert.kernel import Kernel
from simbert.trainers.trainer import Trainer
from simbert.losses.loss import Loss
from simbert.metrics.metric import Metric

log = logging.getLogger(__name__)


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

        self.trainer = None

        self.set_trainer()

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

    def load_dataset(self) -> bool:
        self.train_dataset, self.val_dataset, self.test_dataset = self.DataProcessor.prepare_dataset(self.tokenizer)
        return self.train_dataset is not None and self.val_dataset is not None

    def train_dataset_ready(self) -> bool:
        return self.train_dataset is not None

    def val_dataset_ready(self) -> bool:
        return self.val_dataset is not None

    def test_dataset_ready(self) -> bool:
        return self.test_dataset is not None

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

    def set_trainer(self, trainer=None):
        if trainer is None and type(self.configs.trainer.trainer_name) is not DotMap:
            trainer = self.configs.trainer.get('trainer_name', None)

        T = Trainer()

        trainer_class = T.get(trainer)

        if trainer_class is not None:
            self.trainer = trainer_class(self.configs.trainer).trainer()

    def get_trainer(self) -> Trainer:
        return self.trainer

    def trainer_ready(self) -> bool:
        return self.trainer is not None

    def set_metrics(self, metrics=None):
        if metrics is None:
            metrics = self.configs.get('metrics', {})

        M = Metric()

        for metric in metrics:
            metric_class = M.get(M.get_class_name(metric))

            if metric_class is not None:
                self.metrics.update({metric: metric_class(name=metric)})

    def get_metrics(self) -> dict:
        return self.metrics

    def set_loss_func(self, loss=None):
        if loss is None and type(self.configs.loss.loss_func_name) is not DotMap:
            loss = self.configs.loss.get('loss_func_name', 'CrossEntropyLoss')

        L = Loss()

        loss_class = L.get(loss)

        if loss_class is not None:
            self.loss_func = loss_class().criterion()

    def get_loss_func(self) -> Loss:
        return self.loss_func

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

    def ready_for_training(self) -> bool:
        if not self.train_dataset_ready():

            self.load_dataset()

            if not self.train_dataset_ready():
                raise ValueError("No training dataset is ready for training")

        if not self.val_dataset_ready():
            raise ValueError("No validation dataset is ready for training")

        if not self.trainer_ready():
            raise ValueError("No trainer was found")

        return True

    def ready_for_test(self) -> bool:
        if not self.test_dataset_ready():

            self.load_dataset()

            if not self.test_dataset_ready():
                raise ValueError("No test dataset is ready for testing")

            if not self.trainer_ready():
                raise ValueError("No trainer was found")

        return True

    def fit(self):
        try:
            if self.ready_for_training():
                return self.trainer.fit(self)

        except ValueError as e:
            log.exception(e)

        return None

    def test(self):
        try:
            if self.ready_for_test():
                return self.trainer.test(self)

        except ValueError as e:
            log.exception(e)
