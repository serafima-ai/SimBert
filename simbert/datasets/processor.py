import torch
from dotmap import DotMap
from torch.utils.data import TensorDataset

from simbert.kernel import Kernel
from simbert.datasets.features import FeaturesProcessor
import pandas as pd


class DataProcessor(Kernel):
    FeaturesProcessor = None

    def __init__(self, configs: DotMap = DotMap()):
        self.configs = configs

        if type(self.configs.features.features_processor_name) is not DotMap:
            self.FeaturesProcessor = FeaturesProcessor().get(self.configs.features.features_processor_name)(
                self.configs.features)

    def get_train_examples(self, df):
        """"""
        pass

    def get_dev_examples(self, df):
        """"""
        pass

    def get_test_examples(self, df):
        """"""
        pass

    def get_dataset(self, path='') -> pd.DataFrame:
        if path is '':
            path = self.configs.get('train_dataset', path)
        return pd.read_csv(path, index_col=0)  # path: './ranker/train.csv'

    def prepare_dataset(self, tokenizer):
        """"""
        pass

    def create_tensor_dataset(self, features) -> TensorDataset:
        return TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.long),
                             torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                             torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                             torch.tensor([f.label for f in features], dtype=torch.long))
