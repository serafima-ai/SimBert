from dotmap import DotMap
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers import SquadV2Processor, SquadV1Processor

from simbert.datasets.bert.BertProcessor import BertProcessor


class BertSquadProcessor(BertProcessor):

    def create_examples(self, file, set_type, data_dir=''):

        examples = []

        evaluate = set_type == "test"

        version_2 = self.configs.get('version_2_with_negative', False)

        processor = SquadV2Processor() if version_2 else SquadV1Processor()

        if file != '':

            if not evaluate:
                examples = processor.get_train_examples(data_dir=data_dir, filename=file)

            else:
                examples = processor.get_dev_examples(data_dir=data_dir, filename=file)

        return examples

    def prepare_dataset(self, tokenizer):
        return super().prepare_dataset(tokenizer)

    def prepare_train_dataset(self, tokenizer):

        file = self.get_dataset(type='json')

        examples = self.get_train_examples(file)

        features, dataset = self.FeaturesProcessor.convert_examples_to_features(examples, tokenizer)

        squad_bert_train_dataset, squad_bert_val_dataset, squad_bert_test_dataset = self.__split_dataset(dataset)

        # train loader
        train_sampler = RandomSampler(squad_bert_train_dataset)

        bert_train_dataloader = DataLoader(squad_bert_train_dataset, sampler=train_sampler,
                                           batch_size=self.configs.batch_size)

        # val loader
        val_sampler = RandomSampler(squad_bert_val_dataset)

        bert_val_dataloader = DataLoader(squad_bert_val_dataset, sampler=val_sampler, batch_size=self.configs.batch_size)

        # test loader
        test_sampler = RandomSampler(squad_bert_test_dataset)

        bert_test_dataloader = DataLoader(squad_bert_test_dataset, sampler=test_sampler, batch_size=self.configs.batch_size)

        return bert_train_dataloader, bert_val_dataloader, bert_test_dataloader

    def resolve_test_dataset(self, squad_bert_test_dataloader, tokenizer):

        examples, features = [], []

        if type(self.configs.test_dataset) is not DotMap:

            test_file = self.get_dataset(self.configs.test_dataset, type='json')

            examples = self.get_test_examples(test_file)

            features, test_dataset = self.FeaturesProcessor.convert_examples_to_features(examples, tokenizer, evaluate=True)

            test_sampler = SequentialSampler(test_dataset)

            squad_bert_test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                              batch_size=self.configs.batch_size)

        return squad_bert_test_dataloader, examples, features
