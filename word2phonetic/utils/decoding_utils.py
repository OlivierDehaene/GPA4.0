from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from copy import deepcopy
from itertools import groupby
from tqdm import tqdm

import pandas as pd

from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

EOS_ID = 1


def get_att_mats(translate_model):
    """
    Get's the tensors representing the attentions from a build model.

    The attentions are stored in a dict on the Transformer object while building
    the graph.

    :param translate_model: Transformer object to fetch the attention weights from.
    :return:
    """
    enc_atts = []
    dec_atts = []
    encdec_atts = []

    prefix = 'transformer/body/'
    postfix = '/multihead_attention/dot_product_attention'

    for i in range(translate_model.hparams.num_hidden_layers):
        enc_att = translate_model.attention_weights[
            '%sencoder/layer_%i/self_attention%s' % (prefix, i, postfix)]
        dec_att = translate_model.attention_weights[
            '%sdecoder/layer_%i/self_attention%s' % (prefix, i, postfix)]
        encdec_att = translate_model.attention_weights[
            '%sdecoder/layer_%i/encdec_attention%s' % (prefix, i, postfix)]
        enc_atts.append(enc_att)
        dec_atts.append(dec_att)
        encdec_atts.append(encdec_att)

    selected_heads = [0, 1,
                      2]  # in our experience, the first, the second last and the last layers are the most interpretable
    encdec_att_mats = [tf.squeeze(tf.reduce_sum(encdec_atts[head], axis=1)) for head in selected_heads]
    enc_att_mats = [tf.squeeze(tf.reduce_sum(enc_atts[head], axis=1)) for head in selected_heads]
    dec_att_mats = [tf.squeeze(tf.reduce_sum(dec_atts[head], axis=1)) for head in selected_heads]

    return encdec_att_mats


def build_model(hparams_set, model_name, data_dir, problem_name, beam_size=1, top_beams=1):
    """Build the graph required to featch the attention weights.

    Args:
      hparams_set: HParams set to build the model with.
      model_name: Name of model.
      data_dir: Path to directory contatining training data.
      problem_name: Name of problem.
      beam_size: (Optional) Number of beams to use when decoding a traslation.
          If set to 1 (default) then greedy decoding is used.

    Returns:
      Tuple of (
          inputs: Input placeholder to feed in ids to be translated.
          targets: Targets placeholder to feed to translation when fetching
              attention weights.
          samples: Tensor representing the ids of the translation.
          att_mats: Tensors representing the attention weights.
      )
    """
    hparams = trainer_lib.create_hparams(
        hparams_set, data_dir=data_dir, problem_name=problem_name)
    translate_model = registry.model(model_name)(
        hparams, tf.estimator.ModeKeys.EVAL)

    inputs = tf.placeholder(tf.int32, shape=(None, None, 1, 1), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(None, None, 1, 1), name='targets')
    translate_model({
        'inputs': inputs,
        'targets': targets,
    })

    # Must be called after building the training graph, so that the dict will
    # have been filled with the attention tensors. BUT before creating the
    # interence graph otherwise the dict will be filled with tensors from
    # inside a tf.while_loop from decoding and are marked unfetchable.
    att_mats = get_att_mats(translate_model)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        samples = translate_model.infer({
            'inputs': inputs,
        }, beam_size=beam_size, top_beams=top_beams, alpha=0.6)['outputs']

    return inputs, targets, samples, att_mats


def _encode(str_input, encoder, padding_to=None):
    encoded_input = [encoder['inputs']._token_to_id[c] for c in str_input] + [EOS_ID]
    if padding_to:
        for _ in range(padding_to - len(encoded_input)):
            encoded_input += [0]
    encoded_input = np.reshape(encoded_input, [1, -1, 1, 1])  # Make it 3D.
    return encoded_input


def _decode(integers, encoder):
    return [encoder['inputs']._safe_id_to_token(i) for i in integers if i > 1]


def _char_encode_old(input, padding_to=None, encoding="UTF-8"):
    """
    Transform txt input to int tokens
    +2 is for special tokens ["<EOS>", "<PAD>"]
    + [1] is to add end of sequence "<EOS>" token

    :param input: String input
    :return: [1, -1, 1, 1] Int array
    """
    inp = [c + 2 for c in input.encode(encoding)] + [1]
    if padding_to:
        for _ in range(padding_to - len(inp)):
            inp += [0]
    inp = np.reshape(inp, [1, -1, 1, 1])

    return inp


def _char_decode_old(input):
    """
    Decode token ids to string and removes padding and eos

    :param input: int array
    :return: String
    """

    return [chr(idx - 2) for idx in input if idx > 1]


def _vocab_encode_old(input, vocab, padding_to=None):
    with open(vocab, "r") as f:
        vocab_arr = [l.strip() for l in f.readlines()]

    try:
        inp = [np.where(np.array(vocab_arr) == (c))[0][0] for c in input] + [1]
    except:
        print("Vocab error : {}".format(input))
    if padding_to:
        for _ in range(padding_to - len(inp)):
            inp += [0]
    inp = np.reshape(inp, [1, -1, 1, 1])
    return inp


def _vocab_decode_old(input, vocab):
    with open(vocab, "r") as f:
        vocab_arr = [l.strip() for l in f.readlines()]
    return [vocab_arr[i] for i in input if i > 1]


def _make_prediction_batch(sess, batch_input, input_tensor, input_phon_tensor, output_phon_tensor, att_mats_list,
                           encoder):
    padding_to = len(max(batch_input, key=len)) + 1

    batch_input_tokenized = np.stack([_encode(input, encoder, padding_to).squeeze(0) for input in batch_input], 0)

    batch_phon_tokenized = sess.run(output_phon_tensor, feed_dict={input_tensor: batch_input_tokenized})

    batch_phon = [_decode(phon_tokenized, encoder) for phon_tokenized in batch_phon_tokenized]

    batch_att_mats = sess.run(att_mats_list, feed_dict={input_tensor: batch_input_tokenized,
                                                        input_phon_tensor: np.reshape(batch_phon_tokenized,
                                                                                      [len(batch_input), -1, 1, 1])})

    batch_sum_all_layers = _normalize(np.sum(np.array(batch_att_mats), axis=0))

    return batch_phon, batch_sum_all_layers

def _make_translation_batch(sess, batch_input, input_tensor, output_phon_tensor, top_beams,
                           encoder):
    padding_to = len(max(batch_input, key=len)) + 1

    batch_input_tokenized = np.stack([_encode(input, encoder, padding_to).squeeze(0) for input in batch_input], 0)

    batch_phon_tokenized = sess.run(output_phon_tensor, feed_dict={input_tensor: batch_input_tokenized})

    batch_phon = []
    for phon_tokenized_beam in batch_phon_tokenized:
        if top_beams > 1:
            batch_phon.append(["".join(_decode(phon_tokenized, encoder)) for phon_tokenized in phon_tokenized_beam])
        else:
            batch_phon.append(["".join(_decode(phon_tokenized_beam, encoder))])
    return batch_phon

def g2p_mapping_batch(sess, batch_input, input_tensor, input_phon_tensor, output_phon_tensor, att_mats_list,
                      encoder):
    """
    Predict the phonetic translation of a word using a Transformer model

    :param input: String, word
    :param model_name: Name of the model to serve
    :return: Array[3], [0] input text, [1] phonetic translation, [2] mapping
    """
    # open channel to tensorflow server

    # get phonetic translation and attention matrices
    batch_phon, batch_sum_all_layers = _make_prediction_batch(sess, batch_input, input_tensor, input_phon_tensor,
                                                              output_phon_tensor, att_mats_list, encoder)

    # make prediction
    mapping_batch = [_mapping(input, batch_phon[idx], batch_sum_all_layers[idx, :len(batch_phon[idx]), :len(input)]) for
                     idx, input in enumerate(batch_input)]

    return batch_input, batch_phon, mapping_batch


def _normalize(matrix):
    """
        input: a numpy matrix
        return: matrix with 0 mean and 1 std
    """
    return (matrix - np.mean(matrix)) / (np.std(matrix) + 1e-10)


def _mapping(inp_text, out_text, sum_all_layers):
    # Base threshold
    # fr : 0.75
    # es : 0.4
    if len(out_text) > 4:
        threshold = 0.4
    else:
        threshold = 0

    # While we have too many silent_letters detected
    while (True):
        # Gets the silent_letters indices
        # We consider that a letter is silent if its attention value is below mean attention + threshold * std attention
        try:
            silent_letters_idx = [i for i, idx in enumerate(np.argmax(sum_all_layers, axis=0))
                                  if sum_all_layers[idx, i] < np.mean(sum_all_layers[idx, :])
                                  + threshold * np.std(sum_all_layers[idx, :])]
        except:
            print(inp_text, out_text, sum_all_layers.shape)
        # Reduces threshold if too many silent letters are detected
        # Can happen in french when we have 3 lettres graphemes
        if len(silent_letters_idx) > 1 / 3 * len(inp_text):
            threshold -= 0.1
        else:
            break

    # Creates the phoneme attribution list
    phon_list = np.array(out_text)[np.argmax(sum_all_layers, axis=0)]
    phon_list[silent_letters_idx] = "#"  # "#" is our encoding for silent letters
    phon_list = phon_list.tolist()  # needed for the += just below

    # Checks if all the phonemes are attributed and if they are only present the correct number of time in the list
    # If not, the phoneme is concatenated to its most probable neighbor
    # and the least probable phoneme is replaced by a silent letter (this can happen for small datasets)
    discard_next = False
    for i, phon in enumerate(out_text):
        if phon not in phon_list and not discard_next:
            probable_idx = np.argmax(sum_all_layers[i, :])
            if (i + 1) < len(out_text) and phon_list[probable_idx] == out_text[i + 1]:
                phon_list[probable_idx] = phon + phon_list[probable_idx]
                discard_next = True
            else:
                phon_list[np.argmax(sum_all_layers[i, :])] += phon
        elif discard_next:
            discard_next = False

    # test = np.where(np.array(phon_list) == phon)[0]
    #     if len(test > 1):
    #         phon_list[np.max(test)] = "%"

    ##NOT WORKING PROPERLY

    # Creates the g_p tupple list
    g_p = [(l, phon_list[i]) for i, l in enumerate(inp_text)]

    # Creates the final g_p mapping
    mapping = []
    for phon, letters in groupby(g_p, lambda x: x[1]):
        graph = "".join([letter[0] for letter in letters])
        mapping.append(graph + "~" + phon)

    # return ["".join(inp_text), " ".join(out_text), mapping]
    return mapping


def _dic_add(value, dic):
    if value not in dic:
        dic[value] = 1
    else:
        dic[value] += 1


def _load_model(model_dir, sess):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
        return True
    return False


def _get_unique_words(wordGp):
    uniqueWordList = []
    for word, pred, gpMatch, copy in wordGp:
        if (word, pred) not in uniqueWordList:
            uniqueWordList.append((word, pred))
        else:
            wordGp.remove((word, pred, gpMatch, copy))
    return wordGp


def _generate_word_list(wordGp, gpProg=None):
    if gpProg:
        gpProg = pd.read_csv(gpProg)
        tempList = []
        for i in range(len(gpProg)):
            lesson = gpProg.loc[i]

            for word, pred, gpMatch, copy in wordGp[:]:
                for gp in gpMatch[:]:
                    if gp == lesson["GP"]:
                        gpMatch.remove(gp)
                if len(gpMatch) == 0:
                    tempList.append(((int(lesson["LESSON"])), ("").join(word), ("").join(pred), (".").join(copy),
                                     len(word), len(pred)))

                    wordGp.remove((word, pred, gpMatch, copy))
        for word, pred, gpMatch, copy in wordGp[:]:
            tempList.append((999, ("").join(word), ("").join(pred), (".").join(copy), len(word), len(pred)))

        wordList = pd.DataFrame()
        wordList = wordList.append(tempList, ignore_index=True)
        wordList.columns = [["LESSON", "ORTHOGRAPHY", "PHONOLOGY", "GPMATCH", "N LETTERS", "N PHONEMES"]]
        return wordList
    else:
        tempList = []
        for word, pred, gpMatch, copy in wordGp[:]:
            tempList.append((("").join(word), (" ").join(pred), (".").join(copy), len(word), len(pred)))
        wordList = pd.DataFrame()
        wordList = wordList.append(tempList, ignore_index=True)
        wordList.columns = [["ORTHOGRAPHY", "PHONOLOGY", "GPMATCH", "N LETTERS", "N PHONEMES"]]
        return wordList


def decode_wordList(sess, wordList, input_tensor, input_phon_tensor, output_phon_tensor, att_mats_list, encoder,
                    decode_to_file, gpProg):
    batch_size = 128  # conservative batch size to dodge out of memory issues
    wordCount = len(wordList)

    phon_results = []
    gp_results = []

    n_batch = wordCount // batch_size
    for idx_batch in tqdm(range(n_batch + 1), "GP Matching"):
        try:
            batch = wordList[idx_batch * batch_size:(idx_batch + 1) * batch_size]
        except:
            batch = wordList[n_batch * batch_size:]
        _, batch_phon_results, batch_gp_results = g2p_mapping_batch(sess, batch, input_tensor, input_phon_tensor,
                                                                    output_phon_tensor, att_mats_list, encoder)
        phon_results.extend(batch_phon_results)
        gp_results.extend(batch_gp_results)

    wordGp = list(zip(wordList, phon_results, deepcopy(gp_results), deepcopy(gp_results)))
    wordGp = _get_unique_words(wordGp)
    wordList = _generate_word_list(wordGp, gpProg)

    wordList.to_csv(decode_to_file, encoding="UTF-8")

def prepare_corpus(wordList, phon):
    corpus = {}
    for w, p in zip(wordList, phon):
        if w in corpus:
            corpus[w].append(p.replace(" ", ""))
        else:
            corpus[w] = [p.replace(" ", "")]
    return corpus

def evaluate_corpus(sess, corpus, input_tensor, output_phon_tensor, encoder, top_beams):
    batch_size = 128  # conservative batch size to dodge out of memory issues
    wordList = list(corpus.keys())
    wordCount = len(wordList)

    phon_results = []

    n_batch = wordCount // batch_size
    for idx_batch in tqdm(range(n_batch + 1), "Phon Translation"):
        try:
            batch = wordList[idx_batch * batch_size:(idx_batch + 1) * batch_size]
        except:
            batch = wordList[n_batch * batch_size:]
        batch_phon_results = _make_translation_batch(sess, batch, input_tensor, output_phon_tensor, top_beams,
                           encoder)
        phon_results.extend(batch_phon_results)

    rates = error_rates(corpus, phon_results)
    print("WER : {:.4%} ; PER : {:.4%}".format(rates[0], rates[1]))


def stats(sess, wordList, phon, input_tensor, input_phon_tensor, output_phon_tensor, att_mats_list, encoder):
    batch_size = 128  # conservative batch size to dodge out of memory issues
    wordCount = len(wordList)

    phon_results = []
    gp_results = []

    n_batch = wordCount // batch_size
    for idx_batch in tqdm(range(n_batch + 1), "GP Matching"):
        try:
            batch = wordList[idx_batch * batch_size:(idx_batch + 1) * batch_size]
        except:
            batch = wordList[n_batch * batch_size:]
        _, batch_phon_results, batch_gp_results = g2p_mapping_batch(sess, batch, input_tensor, input_phon_tensor,
                                                                    output_phon_tensor, att_mats_list, encoder)
        gp_results.extend(batch_gp_results)
        phon_results.extend(batch_phon_results)

    phon_results = ["".join(phon) for phon in phon_results]
    gp_results = np.delete(np.array(gp_results), np.where(np.array(phon)!=np.array(phon_results)))

    gpList = []
    gpCountDic = {}
    gCountDic = {}
    pCountDic = {}

    for r in gp_results:
        for gp in r:
            g, p = gp.split("~")
            if gp not in gpList:
                gpList.append(gp)
            _dic_add(gp, gpCountDic)
            _dic_add(g, gCountDic)
            _dic_add(p, pCountDic)

    gpCount = sum(gpCountDic.values())
    tupList = []
    for i in range(len(gpList)):
        gp = gpList[i]
        g, p = gp.split("~")
        tup = (gp, gpCountDic[gp] / gpCount,
               gpCountDic[gp] / gCountDic[g], gpCountDic[gp] / pCountDic[p])
        tupList.append(tup)

    df = pd.DataFrame()
    df = df.append(tupList, ignore_index=True)
    df.columns = [["GP", "GP FREQ IN DATASET", "G CONSISTENCY", "P CONSISTENCY"]]

    weights = [50, 30, 20]
    scores = np.dot(weights, np.transpose(df[["GP FREQ IN DATASET", "G CONSISTENCY", "P CONSISTENCY"]]))
    df["SCORE"] = scores
    df.sort_values(["SCORE"], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    gpProg = df[["GP"]]
    gpProg["LESSON"] = gpProg.index + 1

    return df, gpProg


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def per(gd, pred):
    min_levenshtein = np.inf
    length_min_levenshtein_pronunciation = np.inf
    for beam in pred:
        for pronunciation in gd:
            current = levenshtein(pronunciation, beam)
            if current < min_levenshtein:
                min_levenshtein = current
                length_min_levenshtein_pronunciation = len(pronunciation)
        return min_levenshtein / length_min_levenshtein_pronunciation


def wer(gd, pred):
    for beam in pred:
        if beam in gd:
            return 0
    return 1

def error_rates(corpus, phon_results):
    wordList = list(corpus.keys())
    perS = 0
    werS = 0
    for i, word in tqdm(enumerate(wordList)):
        werS += wer(corpus[word], phon_results[i])
        perS += per(corpus[word], phon_results[i])
    return werS / len(wordList), perS / len(wordList)
