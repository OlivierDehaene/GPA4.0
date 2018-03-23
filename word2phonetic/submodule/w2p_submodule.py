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
from tensor2tensor.utils import registry

import tensorflow as tf

import six
import os

DATASET_FILE = "train_dataset.csv"
CSV_SEPARATOR = ";"


@registry.register_problem
class WordToPhoneticVocab(text_problems.Text2TextProblem):

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    @property
    def vocab_filename(self):
        return "vocab"

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
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        source_path = os.path.join(data_dir, DATASET_FILE)
        with tf.gfile.GFile(source_path, mode="r") as source_file:
            for line in source_file:
                if line and CSV_SEPARATOR in line:
                    parts = line.split(CSV_SEPARATOR, 1)
                    source, target = parts[0].strip(), parts[1].strip()
                    yield {
                        "inputs": " ".join(list(source)),
                        "targets": target
                    }


@registry.register_problem
class WordToPhonetic(text_problems.Text2TextProblem):
    """Predict phonetic from words/pseudo-words"""

    @property
    def vocab_type(self):
        # `ByteTextEncoder`, encode raw bytes.
        return text_problems.VocabType.CHARACTER

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
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        source_path = os.path.join(data_dir, DATASET_FILE)
        with tf.gfile.GFile(source_path, mode="r") as source_file:
            for line in source_file:
                if line and CSV_SEPARATOR in line:
                    parts = line.split(CSV_SEPARATOR, 1)
                    source, target = parts[0].strip(), parts[1].strip()
                    yield {
                        "inputs": source,
                        "targets": target
                    }

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        encoder = LatinByteTextEncoder()
        return encoder

class LatinByteTextEncoder(text_encoder.TextEncoder):
    """Encodes each byte to an id. For 8-bit strings only."""

    def encode(self, s):
        numres = self._num_reserved_ids

        return [c + numres for c in s.encode("Latin-1")]

    def decode(self, ids):
        numres = self._num_reserved_ids
        decoded_ids = []
        int2byte = six.int2byte
        for id_ in ids:
            if 0 <= id_ < numres:
                decoded_ids.append(text_encoder.RESERVED_TOKENS_BYTES[int(id_)])
            else:
                decoded_ids.append(int2byte(id_ - numres))

        return b"".join(decoded_ids).decode("Latin-1", "replace")


@registry.register_hparams
def w2p():
  hparams = transformer.transformer_big_single_gpu()
  hparams.clip_grad_norm = 1.0
  hparams.batch_size = 4500
  return hparams