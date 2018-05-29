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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import tensorflow as tf

from tensor2tensor.utils import usr_dir
from tensor2tensor import problems

from gpa.scripts.decoding_utils import load_model, decode_wordList, build_model

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode_from_file', type=str, required=True)
    parser.add_argument('--decode_to_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gp_prog', type=str, default=None)
    parser.add_argument('--sep', type=str, default=",")
    parser.add_argument('--problem_name', type=str, default="word_to_phonetic")
    parser.add_argument('--model_name', type=str, default="transformer")
    parser.add_argument('--hparams_set', type=str, default="w2p")
    parser.add_argument('--t2t_usr_dir', type=str, default=os.path.join(__location__, "../submodule"))
    args = parser.parse_args()

    with open(args.decode_from_file, 'r') as f:
        wordList = [line.strip().split(args.sep)[0] for line in f.readlines()]

    usr_dir.import_usr_dir(args.t2t_usr_dir)
    input_tensor, input_phon_tensor, output_phon_tensor, encdec_att_mats, enc_att_mats, dec_att_mats = build_model(args.hparams_set, args.model_name,
                                                                                                    args.data_dir, args.problem_name,
                                                                                                    beam_size=5)
    problem = problems.problem(args.problem_name)
    encoder = problem.feature_encoders(args.data_dir)

    sess = tf.Session()

    assert load_model(args.model_dir, sess)

    decode_wordList(sess, wordList, input_tensor, input_phon_tensor, output_phon_tensor, encdec_att_mats, encoder, args.decode_to_file, args.gp_prog)

if __name__ == "__main__":
    tf.app.run()
