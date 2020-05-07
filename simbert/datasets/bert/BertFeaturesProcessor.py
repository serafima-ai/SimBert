from simbert.datasets.features import FeaturesProcessor
from transformers import glue_convert_examples_to_features as convert_examples_to_features


class BertFeaturesProcessor(FeaturesProcessor):

    def convert_examples_to_features(self, examples, tokenizer, output_mode='classification'):
        return convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=self.configs.labels,  # [0, 1]
                                            max_length=self.configs.max_length,  # 368
                                            output_mode=output_mode,
                                            pad_on_left=False,
                                            pad_token=tokenizer.pad_token_id,
                                            pad_token_segment_id=0)
