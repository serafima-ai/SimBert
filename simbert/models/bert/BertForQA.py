from statistics import mean
import pytorch_lightning as pl
from dotmap import DotMap
from torch import nn
from torch.utils.data import DataLoader
from transformers.data.metrics.squad_metrics import normalize_answer
from transformers.data.processors.squad import SquadResult

from simbert.metrics.squad_metrics import compute_predictions_logits
from simbert.models.lightning import SimbertLightningModule
from simbert.models.model import Model
from transformers import *
import torch
from simbert.datasets.processor import DataProcessor


class BertForQA(SimbertLightningModule, Model):

    def __init__(self, configs: DotMap = DotMap(), *args, **kwargs):
        pl.LightningModule.__init__(self, *args, **kwargs)
        Model.__init__(self, configs)

        self.bert = None
        self.num_classes = 0
        self.qa_outputs = None
        self.DataProcessor = None

        self.apply_configs(self.configs)

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
        self.qa_outputs = self.__classifier()

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

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                start_positions=None,
                end_positions=None,
                ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            outputs = (start_logits, end_logits, start_positions, end_positions) + outputs[2:]
        else:
            outputs = (start_logits, end_logits)

        return outputs

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch

        # fwd
        outputs = self.forward(input_ids, attention_mask, token_type_ids,
                               start_positions=start_positions, end_positions=end_positions)

        start_logits, end_logits, start_positions, end_positions = outputs[0], outputs[1], outputs[2], outputs[3]

        start_loss = self.loss_func(start_logits, start_positions)

        end_loss = self.loss_func(end_logits, end_positions)

        loss = (start_loss + end_loss) / 2

        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch

        # fwd
        outputs = self.forward(input_ids, attention_mask, token_type_ids,
                               start_positions=start_positions, end_positions=end_positions)

        start_logits, end_logits, start_positions, end_positions = outputs[0], outputs[1], outputs[2], outputs[3]

        start_loss = self.loss_func(start_logits, start_positions)

        end_loss = self.loss_func(end_logits, end_positions)

        loss = (start_loss + end_loss) / 2

        return {**{'val_loss': loss},
                **self.calculate_metrics(start_logits.cpu(), start_positions.cpu(), stage='val',
                                         apply=lambda x: torch.tensor(x, dtype=torch.float64)),
                **self.calculate_metrics(end_logits.cpu(), end_positions.cpu(), stage='val',
                                         apply=lambda x: torch.tensor(x, dtype=torch.float64))
                }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        avg_metrics = {}

        for _, metric in self.metrics.items():
            key_name = 'val_' + metric.get_metric_name()
            avg_metrics.update({'avg_' + key_name: torch.stack([x[key_name] for x in outputs]).mean()})

        tensorboard_logs = {**{'val_loss': avg_loss}, **avg_metrics}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }

        all_results = []

        outputs = self.forward(**inputs)

        example_indices = batch[3]

        examples = self.test_examples()
        features = self.test_features()

        batch_features = []

        batch_examples = [examples[example_index.item()] for i, example_index in enumerate(example_indices)]

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)
            # print("result", result.unique_id)
            all_results.append(result)

            batch_features.append(eval_feature)

        predictions = compute_predictions_logits(examples, batch_features, all_results, do_lower_case=True,
                                                      version_2_with_negative=True, tokenizer=self.tokenizer)

        answers_data, predictions = get_metrics_input(batch_examples, predictions)

        return {**self.calculate_metrics(predictions, answers_data, stage='test')}

    def test_end(self, outputs):

        avg_metrics = {}

        for _, metric in self.metrics.items():
            key_name = 'test_' + metric.get_metric_name()
            print([x[key_name] for x in outputs])
            avg_metrics.update({'avg_' + key_name: mean([x[key_name] for x in outputs])})

        tensorboard_logs = avg_metrics

        self.test_results = avg_metrics

        return {**avg_metrics, **{'log': tensorboard_logs, 'progress_bar': tensorboard_logs}}

    @pl.data_loader
    def train_dataloader(self):
        return self.train_dataset

    @pl.data_loader
    def val_dataloader(self):
        return self.val_dataset

    @pl.data_loader
    def test_dataloader(self):
        if len(self.test_dataset) == 3:
            return self.test_dataset[0]
        return self.test_dataset

    def test_examples(self):
        if len(self.test_dataset) == 3:
            return self.test_dataset[1]
        return None

    def test_features(self):
        if len(self.test_dataset) == 3:
            return self.test_dataset[2]
        return None

    def train_model(self):
        pass

    def evaluate_model(self):
        pass


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def get_metrics_input(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    """"""

    pairs_answers = []
    pairs_predictions = []

    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]

        pairs_answers.append(([normalize_answer(a) if a else prediction for a in gold_answers], qas_id))
        pairs_predictions.append(normalize_answer(prediction) if prediction else prediction)

    answers_data = [pairs_answers, qas_id_to_has_answer, has_answer_qids, no_answer_qids, no_answer_probs,
                    no_answer_probability_threshold]

    return answers_data, pairs_predictions


