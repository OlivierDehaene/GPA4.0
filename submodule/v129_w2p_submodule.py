# coding=utf-8
# Copyright 2018 Olivier Dehaene
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

from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import problem
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import tensorflow as tf

import six
import os

EOS = text_encoder.EOS_ID

@registry.register_problem
class WordToPhoneticVocab(problem.Text2TextProblem):
    @property
    def targeted_vocab_size(self):
        return 2 ** 7  # 128

    @property
    def is_character_level(self):
        return False

    @property
    def use_subword_tokenizer(self):
        return False

    @property
    def num_shards(self):
        return 10

    @property
    def vocab_name(self):
        return "vocab"

    @property
    def vocab_file(self):
        return self.vocab_name

    def generator(self, data_dir, tmp_dir, train):
        token_vocab = text_encoder.TokenTextEncoder(os.path.join(data_dir, self.vocab_name))
        data_path = os.path.join(data_dir, 'train_dataset.csv')
        return csv_generator(data_path, token_vocab, sep=";", eos=EOS, split_source=True)

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC


@registry.register_problem
class WordToPhoneticChr(translate.TranslateProblem):
    @property
    def is_character_level(self):
        return True

    @property
    def vocab_name(self):
        return "vocab"

    def generator(self, data_dir, tmp_dir, train):
        character_vocab = LatinByteTextEncoder()
        data_path = os.path.join(data_dir, 'train_dataset.csv')
        return csv_generator(data_path, character_vocab, sep=";", eos=EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_CHR

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC


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


def csv_generator(source_path, encoder, sep=";", eos=None, split_source=False, split_target=False):
    """Generator for sequence-to-sequence tasks using tabbed files.

    Tokens are derived from text files where each line contains both
    a source and a target string. The two strings are separated by a separator
    character. It yields dictionaries of "inputs" and "targets" where
    inputs are characters from the source lines converted to integers, and
    targets are characters from the target lines, also converted to integers.

    Args:
      source_path: path to the file with source and target sentences.
      encoder: a TextEncoder to encode the source and target strings.
      sep: a separator character
      eos: integer to append at the end of each sequence (default: None).
      split_source: boolean to split the source string.
                    Needed for TokenTextEncoder if we want to emulate char encoders behaviour.
      split_target: boolean to split the target string.
                    Needed for TokenTextEncoder if we want to emulate char encoders behaviour.
    Yields:
      A dictionary {"inputs": source-line, "targets": target-line} where
      the lines are integer lists converted from characters in the file lines.
    """
    eos_list = [] if eos is None else [eos]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        for line in source_file:
            if line and sep in line:
                parts = line.split(sep, 1)
                source, target = parts[0].strip(), parts[1].strip()
                if split_source:
                    source = " ".join(list(source))
                if split_target:
                    target = " ".join(list(target))
                source_ints = encoder.encode(source) + eos_list
                target_ints = encoder.encode(target) + eos_list
                yield {"inputs": source_ints, "targets": target_ints}


@registry.register_hparams
def w2p():
    hparams = transformer.transformer_base_single_gpu()
    hparams.clip_grad_norm = 1.0
    hparams.batch_size = 14000 # Reduce this number if you have Out Of Memory (OOM) errors. Works on a 1080 TI
    return hparams
