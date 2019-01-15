# coding=utf-8
# Copyright 2018 Olivier Dehaene.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.models import transformer
from tensor2tensor.models.research import universal_transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import mlperf_log

import tensorflow as tf

import six
import os

DATASET_FILE = "train_dataset.csv"


@registry.register_problem
class GraphemeToPhoneme(text_problems.Text2TextProblem):

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    @property
    def vocab_filename(self):
        return "vocab"

    def feature_encoders(self, data_dir):
        encoders = {"targets": self.get_or_create_vocab(data_dir, is_target=True)}
        if self.has_inputs:
            encoders["inputs"] = self.get_or_create_vocab(data_dir, is_target=False)
        return encoders

    def get_or_create_vocab(self, data_dir, is_target):
        assert self.vocab_type == text_problems.VocabType.TOKEN
        vocab_filename = 'source_' + self.vocab_filename if not is_target else 'target_' + self.vocab_filename
        vocab_filename = os.path.join(data_dir, vocab_filename)
        encoder = text_encoder.TokenTextEncoder(vocab_filename,
                                                replace_oov=self.oov_token)

        return encoder

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
        elif dataset_split == problem.DatasetSplit.EVAL:
            mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)

        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_or_create_vocab(data_dir, is_target=False)
        target_encoder = self.get_or_create_vocab(data_dir, is_target=True)
        return text2text_generate_encoded(generator, encoder,
                                          has_inputs=self.has_inputs,
                                          targets_vocab=target_encoder)

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        },
            {
                "split": problem.DatasetSplit.EVAL,
                "shards": 1,
            }
        ]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        source_path = os.path.join(data_dir, DATASET_FILE)
        with tf.gfile.GFile(source_path, mode="r") as source_file:
            for line in source_file:
                if line:
                    csv_sep = ";"
                    if csv_sep not in line:
                        csv_sep = ","
                    parts = line.split(csv_sep, 1)
                    source, target = parts[0].strip(), parts[1].strip()
                    yield {
                        "inputs": " ".join(list(source)),
                        "targets": target
                    }


def text2text_generate_encoded(sample_generator,
                               vocab,
                               targets_vocab,
                               has_inputs=True):
    """Encode Text2Text samples from the generator with the vocab."""
    for sample in sample_generator:
        if has_inputs:
            sample["inputs"] = vocab.encode(sample["inputs"])
            sample["inputs"].append(text_encoder.EOS_ID)
        sample["targets"] = targets_vocab.encode(sample["targets"])
        sample["targets"].append(text_encoder.EOS_ID)
        yield sample


@registry.register_hparams
def g2p():
    hparams = transformer.transformer_base_single_gpu()
    hparams.length_bucket_step = 1.5
    hparams.max_length = 30
    hparams.min_length_bucket = 6
    hparams.keep_checkpoint_max = 10

    hparams.batch_size = 20000
    hparams.num_heads = 8
    hparams.filter_size = 512
    hparams.hidden_size = 256
    hparams.num_hidden_layers = 4
    hparams.self_attention_type = 'dot_product_relative_v2'
    hparams.max_relative_position = 4
    hparams.layer_prepostprocess_dropout = 0.3
    hparams.attention_dropout = 0.2
    return hparams


@registry.register_hparams
def g2p_smaller():
    hparams = transformer.transformer_base_single_gpu()
    hparams.length_bucket_step = 1.5
    hparams.max_length = 30
    hparams.min_length_bucket = 4
    hparams.keep_checkpoint_max = 10

    hparams.batch_size = 20000
    hparams.num_heads = 4
    hparams.filter_size = 512
    hparams.hidden_size = 256
    hparams.num_hidden_layers = 3
    hparams.self_attention_type = 'dot_product_relative_v2'
    hparams.max_relative_position = 4
    hparams.layer_prepostprocess_dropout = 0.3
    hparams.attention_dropout = 0.2
    return hparams


@registry.register_hparams
def g2p_old():
    hparams = transformer.transformer_base_single_gpu()
    hparams.length_bucket_step = 1.5
    hparams.max_length = 30
    hparams.min_length_bucket = 6
    hparams.keep_checkpoint_max = 10

    hparams.batch_size = 20000
    hparams.num_heads = 4
    hparams.filter_size = 512
    hparams.hidden_size = 256
    hparams.num_hidden_layers = 3
    hparams.layer_prepostprocess_dropout = 0.3
    hparams.attention_dropout = 0.2
    return hparams


@registry.register_hparams
def g2p_universal():
    hparams = universal_transformer.universal_transformer_base()
    hparams.length_bucket_step = 1.5
    hparams.max_length = 30
    hparams.min_length_bucket = 6
    hparams.keep_checkpoint_max = 10

    hparams.batch_size = 20000
    hparams.num_heads = 4
    hparams.filter_size = 512
    hparams.hidden_size = 256
    hparams.num_hidden_layers = 3
    hparams.layer_prepostprocess_dropout = 0.3
    hparams.attention_dropout = 0.2
    return hparams
