from dotmap import DotMap
from transformers import BertTokenizer, BertForQuestionAnswering
from simbert.models.transformers import TransformerModel
from simbert.models.model import Model
import torch


class BertForQA(Model, TransformerModel):

    def __init__(self, configs: DotMap = None):
        super().__init__(configs)
        self.tokenizer = BertTokenizer.from_pretrained(
            self.configs.get('bert_tokenizer', 'bert-base-multilingual-cased'))  # e.g. bert-base-multilingual-cased
        self.model = BertForQuestionAnswering.from_pretrained(
            self.configs.get('bert_model', 'distilbert-base-cased-distilled-squad'))  # path to pretrained model

    def predict(self, inputs: []) -> dict:
        question, context = inputs[0]

        input_ids = self.tokenizer.encode(question, context)

        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]

        start_scores, end_scores = self.model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        start_score, end_score = torch.argmax(start_scores), torch.argmax(end_scores)

        answer = ' '.join(all_tokens[start_score: end_score + 1])

        return {'answer': answer, 'proba': {'start_score': start_score, 'end_score': end_score}}

    def train(self):
        self.train_transformer(self.configs, model=self.model, tokenizer=self.tokenizer)

    def evaluate(self):
        self.evaluate_transformer(self.configs, model=self.model, tokenizer=self.tokenizer)

    def test(self):
        pass


