from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import argparse
import os
import sys

from sklearn.model_selection import train_test_split


def train_test_split_dataset(words, phonetic_words, output_dir, test_size=0.25, seed=123):
    txt_train, txt_test, phon_train, phon_test = train_test_split(np.array(words), np.array(phonetic_words),
                                                                  test_size=test_size, random_state=seed)

    train_dataset = pd.DataFrame({"Input": txt_train, "Phon": phon_train})
    train_dataset.to_csv(os.path.join(output_dir, "train_dataset.csv"), header=False, index=False)

    test_dataset = pd.DataFrame({"Input": txt_test, "Phon": phon_test})
    test_dataset.to_csv(os.path.join(output_dir, "test_dataset.csv"), header=False, index=False)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--csv_sep', type=str, default=",")
    parser.add_argument('--spelling_column', type=str, default="ORTHO")
    parser.add_argument('--phon_column', type=str, default="PHON")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path, sep=args.csv_sep)

    words = df[args.spelling_column]
    phonetic_words = df[args.phon_column]

    train_test_split_dataset(words, phonetic_words, args.output_dir, args.test_size, args.seed)


if __name__ == '__main__':
    main(sys.argv)
