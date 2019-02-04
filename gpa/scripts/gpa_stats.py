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
import pandas as pd

from tensor2tensor.utils import usr_dir
from tensor2tensor import problems
import warnings

warnings.filterwarnings('ignore')

from gpa.scripts.decoding_utils import load_model, build_model, stats

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--csv_sep', type=str, default=",")
    parser.add_argument('--problem_name', type=str, default="grapheme_to_phoneme")
    parser.add_argument('--model_name', type=str, default="transformer")
    parser.add_argument('--hparams_set', type=str, default="g2p")
    parser.add_argument('--t2t_usr_dir', type=str, default=os.path.join(__location__, "../submodule"))
    parser.add_argument('--weights', type=list, default=[100, 0, 0])
    # parser.add_argument('--freq_column', type=list, default=[50, 30, 20])
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep=args.csv_sep)
    wordList = df.iloc[:, 0]
    phon = df.iloc[:, 1]

    usr_dir.import_usr_dir(args.t2t_usr_dir)
    input_tensor, input_phon_tensor, output_phon_tensor, encdec_att_mats = build_model(
        args.hparams_set, args.model_name,
        args.data_dir, args.problem_name,
        beam_size=1)
    problem = problems.problem(args.problem_name)
    encoder = problem.feature_encoders(args.data_dir)

    sess = tf.Session()

    assert load_model(args.model_dir, sess)

    rstats, gpProg = stats(sess, wordList, phon, input_tensor, input_phon_tensor, output_phon_tensor, encdec_att_mats,
                           encoder, args.weights)

    rstats.to_csv(os.path.join(args.output_dir, "stats.csv"))
    gpProg.to_csv(os.path.join(args.output_dir, "gpProg.csv"), index=False)


if __name__ == "__main__":
    tf.app.run()
