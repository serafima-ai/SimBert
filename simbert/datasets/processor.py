import torch
from torch.utils.data import TensorDataset

from simbert.kernel import Kernel
from simbert.datasets.features import FeaturesProcessor
import pandas as pd


class DataProcessor(Kernel):
    configs = None

    FeaturesProcessor = None

    def __init__(self, configs=None):
        if configs is None:
            configs = {}
        self.configs = configs
        self.FeaturesProcessor = FeaturesProcessor().get(configs.features.processor_name)(configs.features)

    def get_train_examples(self, df):
        pass

    def get_dev_examples(self, df):
        pass

    def get_dataset(self, path='') -> pd.DataFrame:
        path = self.configs.get('train_path', path)
        return pd.read_csv(path, index_col=0)  # path: './ranker/train.csv'

    def prepare_dataset(self, tokenizer):
        pass

    def __create_tensor_dataset(self, features) -> TensorDataset:
        return TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.long),
                             torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                             torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                             torch.tensor([f.label for f in features], dtype=torch.long))
