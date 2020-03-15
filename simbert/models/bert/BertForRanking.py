import pytorch_lightning as pl
from torch import nn

from simbert.models.model import Model
from transformers import *
import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


class BertForRanking(Model, pl.LightningModule):
    # y = это преобразованный label для расчета под binary_cross_entropy_with_logits,
    # для cross_entropy ставишь label в loss функцию
    def __init__(self, configs: dict):
        super(BertForRanking, self).__init__()

        self.bert = BertModel.from_pretrained('D:/tmp/debug_squad')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_ids, attention_mask, token_type_ids):

        h, _, attn = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)

        h_cls = h[:, 0]
        logits = self.classifier(h_cls)
        sigmoids = self.sigmoid(logits)

        return sigmoids, attn

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # fwd
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        y = torch.zeros(label.shape[0], 2, device=' cuda')
        y[range(y.shape[0]), label] = 1

        # loss
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        # loss = F.cross_entropy(y_hat, label)
        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # fwd
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        y = torch.zeros(label.shape[0], 2, device=' cuda')
        y[range(y.shape[0]), label] = 1
        # print(y_hat,'label',label,'new',y)

        # loss
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        # print(loss)
        # loss = F.cross_entropy(y_hat, label)
        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch

        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())

        return {'test_acc': torch.tensor(test_acc)}

    def test_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    @pl.data_loader
    def train_dataloader(self):
        return bert_rank_train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return bert_rank_val_dataloader