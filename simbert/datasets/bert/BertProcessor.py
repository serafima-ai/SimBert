from torch.utils.data import random_split, DataLoader, RandomSampler
from transformers.data.processors.utils import InputExample
from simbert.datasets.processor import DataProcessor


class BertProcessor(DataProcessor):
    """Dataset processor for BERT-based models."""

    def get_train_examples(self, df):
        """See base class."""
        return self.__create_examples(df, "train")

    def get_dev_examples(self, df):
        """See base class."""
        return self.__create_examples(df, "dev_matched")

    def __create_examples(self, df, set_type):
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

        dataframe = self.get_dataset()

        dataset = self.get_train_examples(dataframe)

        features = self.FeaturesProcessor.convert_examples_to_features(dataset, tokenizer)

        train_dataset = self.__create_tensor_dataset(features)

        bert_train_dataset, bert_val_dataset = self.__split_dataset(train_dataset)

        # train loader
        train_sampler = RandomSampler(bert_train_dataset)

        bert_train_dataloader = DataLoader(bert_train_dataset, sampler=train_sampler,
                                           batch_size=self.configs.batch_size)

        # val loader
        val_sampler = RandomSampler(bert_val_dataset)

        bert_val_dataloader = DataLoader(bert_val_dataset, sampler=val_sampler, batch_size=self.configs.batch_size)

        return bert_train_dataloader, bert_val_dataloader

    def __split_dataset(self, train_dataset, train_set=0.87):
        train_set_split_prop = float(self.configs.get('train_set_proportion', train_set))

        nb_train_samples = int(train_set_split_prop * len(train_dataset))

        nb_val_samples = len(train_dataset) - nb_train_samples

        return random_split(train_dataset, [nb_train_samples, nb_val_samples])
