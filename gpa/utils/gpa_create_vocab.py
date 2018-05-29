from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import argparse
import collections
import io
import os


def create_vocab(dataframe, output_dir, phoneme_separator):
    cnt = collections.Counter()

    for word in dataframe["ORTHO"]:
        word = str(word)
        tokens = list(word.strip())  # Emulates char encoding

        tokens = [_ for _ in tokens if len(_) > 0]
        cnt.update(tokens)

    for phonetic_word in dataframe["PHON"]:
        phonetic_word = str(phonetic_word)
        if phoneme_separator != "":
            tokens = phonetic_word.strip().split(phoneme_separator)
        else:
            tokens = list(phonetic_word.strip())

        tokens = [_ for _ in tokens if len(_) > 0]
        cnt.update(tokens)

    print("Found " + str(len(cnt)) + " unique tokens in the vocabulary.")

    token_with_counts = cnt.most_common()
    token_with_counts = sorted(
        token_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

    with io.open(os.path.join(output_dir, "vocab"), 'w', encoding='utf-8') as f:
        f.write("<pad>\n")  # 0 is reserved for the pad token
        f.write("<EOS>\n")  # 1 is reserved for the end of sequence token
        for token, count in token_with_counts:
            f.write(token + "\n")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--csv_sep', type=str, default=",")
    parser.add_argument('--phoneme_sep', type=str, default="")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path, sep=args.csv_sep, header=None)
    df.columns = ["ORTHO", "PHON"]

    create_vocab(df, args.output_dir, args.phoneme_sep)