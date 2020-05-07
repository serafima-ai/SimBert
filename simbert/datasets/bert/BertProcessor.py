from dotmap import DotMap
from torch.utils.data import random_split, DataLoader, RandomSampler, TensorDataset
from transformers.data.processors.utils import InputExample
from simbert.datasets.processor import DataProcessor


class BertProcessor(DataProcessor):
    """Dataset processor for BERT-based models."""

    def get_train_examples(self, df):
        """See base class."""
        return self.create_examples(df, "train")

    def get_dev_examples(self, df):
        """See base class."""
        return self.create_examples(df, "dev_matched")

    def get_test_examples(self, df):
        """See base class."""
        return self.create_examples(df, "test")

    def create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx, row in df.iterrows():
            guid = "%s-%s" % (set_type, idx)

            default_input_columns = ['query', 'text']
            default_output_columns = ['label']

            input_columns = self.configs.get('input', default_input_columns)
            output_columns = self.configs.get('output', default_output_columns)

            try:
                text_a, text_b = row[input_columns[0]], None
                if len(input_columns) > 1:
                    text_b = row[input_columns[1]]

                label = row[output_columns[0]]

            except KeyError:
                print('No corresponding columns found for config keys {}'.format(input_columns))

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def prepare_dataset(self, tokenizer):

        bert_train_dataloader, bert_val_dataloader, bert_test_dataloader = None, None, None

        if type(self.configs.train_dataset) is not DotMap:
            bert_train_dataloader, bert_val_dataloader, bert_test_dataloader = self.prepare_train_dataset(tokenizer)

        bert_test_dataloader = self.resolve_test_dataset(bert_test_dataloader, tokenizer)

        return bert_train_dataloader, bert_val_dataloader, bert_test_dataloader

    def prepare_train_dataset(self, tokenizer):

        dataframe = self.get_dataset()

        dataset = self.get_train_examples(dataframe)

        features = self.FeaturesProcessor.convert_examples_to_features(dataset, tokenizer)

        train_dataset = self.create_tensor_dataset(features)

        bert_train_dataset, bert_val_dataset, bert_test_dataset = self.__split_dataset(train_dataset)

        # train loader
        train_sampler = RandomSampler(bert_train_dataset)

        bert_train_dataloader = DataLoader(bert_train_dataset, sampler=train_sampler,
                                           batch_size=self.configs.batch_size)

        # val loader
        val_sampler = RandomSampler(bert_val_dataset)

        bert_val_dataloader = DataLoader(bert_val_dataset, sampler=val_sampler, batch_size=self.configs.batch_size)

        # test loader
        test_sampler = RandomSampler(bert_test_dataset)

        bert_test_dataloader = DataLoader(bert_test_dataset, sampler=test_sampler, batch_size=self.configs.batch_size)

        return bert_train_dataloader, bert_val_dataloader, bert_test_dataloader

    def resolve_test_dataset(self, bert_test_dataloader, tokenizer):
        if type(self.configs.test_dataset) is not DotMap:
            test_dataframe = self.get_dataset(self.configs.test_dataset)

            dataset = self.get_test_examples(test_dataframe)

            features = self.FeaturesProcessor.convert_examples_to_features(dataset, tokenizer)

            test_dataset = self.create_tensor_dataset(features)

            test_sampler = RandomSampler(test_dataset)

            bert_test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                              batch_size=self.configs.batch_size)

        return bert_test_dataloader

    def __split_dataset(self, train_dataset, train_set=0.87, val_set=0.13, test_set=0.0):
        train_set_split_prop = float(self.configs.get('train_set_proportion', train_set))

        val_set_split_prop = float(self.configs.get('val_set_proportion', val_set))

        test_set_split_prop = float(self.configs.get('test_set_proportion', test_set))

        if train_set_split_prop + val_set_split_prop > 1 and test_set_split_prop == 0:
            val_set_split_prop = 1 - train_set_split_prop

        if train_set_split_prop + val_set_split_prop + test_set_split_prop != 1:
            raise ValueError("Train, validation, test set proportions are not equal to 1")

        nb_train_samples = int(train_set_split_prop * len(train_dataset))

        nb_val_samples = int(val_set_split_prop * len(train_dataset))  # len(train_dataset) - nb_train_samples

        nb_test_samples = len(train_dataset) - nb_train_samples - nb_val_samples

        if nb_test_samples == 0:
            return random_split(train_dataset, [nb_train_samples, nb_val_samples]), TensorDataset()

        return random_split(train_dataset, [nb_train_samples, nb_val_samples, nb_test_samples])
