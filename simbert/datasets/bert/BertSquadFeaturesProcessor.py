from simbert.datasets.features import FeaturesProcessor
from transformers import squad_convert_examples_to_features


class BertSquadFeaturesProcessor(FeaturesProcessor):

    def convert_examples_to_features(self, examples, tokenizer, output_mode='', evaluate=False):

        max_seq_length = self.configs.get('max_seq_length', 384)
        doc_stride = self.configs.get('doc_stride', 128)
        max_query_length = self.configs.get('max_query_length', 64)
        threads = self.configs.get('num_threads', 1)

        return squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=threads,
        )


