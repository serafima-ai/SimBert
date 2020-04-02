import pytorch_lightning as pl
from dotmap import DotMap
from torch import nn
from torch.utils.data import DataLoader

from simbert.models.lightning import SimbertLightningModule
from simbert.models.model import Model
from transformers import *
import torch
from sklearn.metrics import accuracy_score
from simbert.datasets.processor import DataProcessor
from simbert.optimizers.optimizer import Optimizer


class BertForRanking(Model, SimbertLightningModule):

    def __init__(self, configs: DotMap = DotMap(), *args, **kwargs):
        pl.LightningModule.__init__(self, *args, **kwargs)
        Model.__init__(self, configs)

        self.bert = None
        self.num_classes = 0
        self.classifier = None
        self.DataProcessor = None

        self.apply_configs(self.configs)

        self.sigmoid = nn.Sigmoid()

        if configs is not None:
            self.DataProcessor = self.data_processor()

    def __bert_model(self):
        if self.bert is not None and self.configs.get('bert_model') is None:
            return self.bert
        return BertModel.from_pretrained(self.configs.get('bert_model', 'bert-base-multilingual-cased'))

    def __calculate_classes(self):
        if self.num_classes != 0 and self.configs.dataset.processor.features.get('labels') is None:
            return self.num_classes
        return len(self.configs.dataset.processor.features.labels)

    def __classifier(self, num_classes=2):
        num_classes = self.configs.get(num_classes, num_classes)
        return nn.Linear(self.bert.config.hidden_size, num_classes)

    def data_processor(self):
        if self.DataProcessor is not None and self.configs.dataset.get(
                'processor') is None or self.configs.dataset.processor.get('data_processor_name') is None:
            return self.DataProcessor
        return DataProcessor().get(self.configs.dataset.processor.data_processor_name)(
            self.configs.dataset.processor)

    def new_tokenizer(self):
        if self.tokenizer is not None and self.configs.get('tokenizer') is None:
            return self.tokenizer
        return BertTokenizer.from_pretrained(
            self.configs.get('tokenizer', 'bert-base-multilingual-cased'))

    def apply_configs(self, configs: DotMap):
        Model.apply_configs(self, configs)

        self.bert = self.__bert_model()
        self.num_classes = self.__calculate_classes()
        self.classifier = self.__classifier()

    def predict(self, inputs):

        examples = []

        results = []

        for sample in inputs:
            query, paragraph = sample

            examples.append(InputExample(text_a=query, text_b=paragraph, label=0, guid='prediction'))

        features = self.DataProcessor.FeaturesProcessor.convert_examples_to_features(examples,
                                                                                     tokenizer=self.tokenizer)

        tokenized = self.DataProcessor.create_tensor_dataset(features)

        bert_test_dataloader = DataLoader(tokenized)

        for batch in bert_test_dataloader:
            input_ids, attention_mask, token_type_ids, label = batch

            results.append(
                self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0][
                    0].tolist())

        return results

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooler_output, attn = outputs[1], outputs[-1]

        logits = self.classifier(pooler_output)

        sigmoids = self.sigmoid(logits)

        return sigmoids, attn

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # fwd
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        y = torch.zeros(label.shape[0], 2, device='cuda')
        y[range(y.shape[0]), label] = 1

        # loss
        # loss = F.binary_cross_entropy_with_logits(y_hat, y)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(y_hat, label)
        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # fwd
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        y = torch.zeros(label.shape[0], 2, device='cuda')
        y[range(y.shape[0]), label] = 1
        # print(y_hat,'label',label,'new',y)

        # loss
        # loss = F.binary_cross_entropy_with_logits(y_hat, y)
        # print(loss)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(y_hat, label)
        # acc
        a, y_hat = torch.max(y_hat, dim=1)

        return {**{'val_loss': loss}, **self.calculate_metrics(label.cpu(), y_hat.cpu(), stage='val',
                                                               apply=lambda x: torch.tensor(x, dtype=torch.float64))}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        avg_metrics = {}

        for _, metric in self.metrics.items():
            key_name = 'val_' + metric.get_metric_name()
            avg_metrics.update({'avg_' + key_name: torch.stack([x[key_name] for x in outputs]).mean()})

        tensorboard_logs = {**{'val_loss': avg_loss}, **avg_metrics}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch

        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        a, y_hat = torch.max(y_hat, dim=1)

        return self.calculate_metrics(label.cpu(), y_hat.cpu(), stage='test',
                                      apply=lambda x: torch.tensor(x, dtype=torch.float64))

    def test_end(self, outputs):

        avg_metrics = {}

        for _, metric in self.metrics.items():
            key_name = 'test_' + metric.get_metric_name()
            avg_metrics.update({'avg_' + key_name: torch.stack([x[key_name] for x in outputs]).mean()})

        tensorboard_logs = avg_metrics

        return {**avg_metrics, **{'log': tensorboard_logs, 'progress_bar': tensorboard_logs}}

    def configure_optimizers(self):
        return Optimizer().get(self.configs.optimizer.optimizer_name)(self.configs.optimizer).optimizer(
            [p for p in self.parameters() if p.requires_grad])

    @pl.data_loader
    def train_dataloader(self):
        return self.train_dataset

    @pl.data_loader
    def val_dataloader(self):
        return self.val_dataset

    @pl.data_loader
    def test_dataloader(self):
        return self.test_dataset

    def train_model(self):
        pass

    def evaluate_model(self):
        pass
