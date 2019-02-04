from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from copy import deepcopy, copy
from itertools import groupby, chain
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
    encdec_atts = []

    prefix = 'transformer/body/'
    postfix = '/multihead_attention/dot_product_attention'

    for i in range(1, translate_model.hparams.num_hidden_layers):
        encdec_att = translate_model.attention_weights[
            '%sdecoder/layer_%i/encdec_attention%s' % (prefix, i, postfix)]
        encdec_atts.append(encdec_att)

    encdec_att_mats = [tf.squeeze(tf.reduce_sum(mat, axis=1)) for mat in encdec_atts]

    return encdec_att_mats


def build_model(hparams_set, model_name, data_dir, problem_name, beam_size=1, top_beams=1):
    """Build the graph required to featch the attention weights.

    Args:
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
    encdec_att_mats = get_att_mats(translate_model)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        samples = translate_model.infer({
            'inputs': inputs,
        }, beam_size=beam_size, top_beams=top_beams, alpha=0.6)['outputs']

    return inputs, targets, samples, encdec_att_mats


def _encode(str_input, encoder, padding_to=None):
    encoded_input = [encoder['inputs']._token_to_id[c] for c in str_input] + [EOS_ID]
    if padding_to:
        for _ in range(padding_to - len(encoded_input)):
            encoded_input += [0]
    encoded_input = np.reshape(encoded_input, [1, -1, 1, 1])  # Make it 3D.
    return encoded_input


def _decode(integers, encoder):
    decoded_str = []
    for i in integers:
        if i == 1:
            break
        elif i != 0:
            decoded_str.append(encoder['targets']._safe_id_to_token(i))
    return decoded_str


def _make_prediction_batch(sess, batch_input, input_tensor, input_phon_tensor, output_phon_tensor, att_mats_list,
                           encoder):
    padding_to = len(max(batch_input, key=len)) + 1

    batch_input_tokenized = np.stack([_encode(input, encoder, padding_to).squeeze(0) for input in batch_input], 0)

    batch_phon_tokenized = sess.run(output_phon_tensor, feed_dict={input_tensor: batch_input_tokenized})

    batch_phon = [_decode(np.squeeze(phon_tokenized), encoder) for phon_tokenized in batch_phon_tokenized]

    batch_att_mats = sess.run(att_mats_list, feed_dict={input_tensor: batch_input_tokenized,
                                                        input_phon_tensor: np.reshape(batch_phon_tokenized,
                                                                                      [len(batch_input), -1, 1, 1])})

    batch_sum_all_layers = np.sum(np.stack(batch_att_mats), axis=0)

    return batch_phon, batch_sum_all_layers


def _make_translation_batch(sess, batch_input, input_tensor, output_phon_tensor, top_beams,
                            encoder):
    padding_to = len(max(batch_input, key=len)) + 1

    batch_input_tokenized = np.stack([_encode(input, encoder, padding_to).squeeze(0) for input in batch_input], 0)

    batch_phon_tokenized = sess.run(output_phon_tensor, feed_dict={input_tensor: batch_input_tokenized})

    batch_phon = []
    for phon_tokenized_beam in batch_phon_tokenized:
        if top_beams > 1:
            batch_phon.append(
                ["".join(_decode(np.squeeze(phon_tokenized), encoder)) for phon_tokenized in phon_tokenized_beam])
        else:
            batch_phon.append(["".join(_decode(np.squeeze(phon_tokenized_beam), encoder))])
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


def reccurent_aggregation(graph, associated_phons):
    for i, phons in enumerate(associated_phons):
        try:
            for p in phons:
                if p in associated_phons[i + 1]:
                    graph[i] = graph[i] + graph.pop(i + 1)
                    associated_phons[i] = associated_phons[i] + list(
                        set(associated_phons.pop(i + 1)) - set(associated_phons[i]))
                    associated_phons[i].sort()
                    return reccurent_aggregation(graph, associated_phons)
        except:
            return graph, associated_phons


def _mapping(inp_text, out_text, sum_all_layers):
    sum_all_layers = sum_all_layers / 8.0
    n_letters = len(inp_text)
    n_phon = len(out_text)

    associated_phons = []

    for i in range(n_letters):
        att_slice = copy(sum_all_layers[:, i])
        max_value = np.max(att_slice)
        masked_att_slice = att_slice[att_slice > 0.25]
        sorted_att_slice_idx = att_slice.argsort()[::-1]
        att_slice.sort()
        sorted_att_slice_values = att_slice[::-1]
        top_values = [idx for idx, v in zip(sorted_att_slice_idx, sorted_att_slice_values) if
                      v > 0.25 and v > (max_value - max(1.5 * np.std(masked_att_slice), 0.1))]
        if not top_values:
            top_values = [n_phon + i]
        associated_phons.append(top_values)

    for i in range(n_phon):
        if i not in list(chain.from_iterable(associated_phons)):
            att_slice = sum_all_layers[i, :]
            idx = np.argmax(att_slice)
            associated_phons[idx] = associated_phons[idx] + [i]

    graph = [[i] for i in range(n_letters)]

    graph, associated_phons = reccurent_aggregation(graph, associated_phons)
    assert len(graph) == len(associated_phons)

    mapping = []
    for g, p in zip(graph, associated_phons):
        str_g = "".join([inp_text[i] for i in g])
        p_chs = []
        for i in p:
            try:
                p_chs.append(out_text[i])
            except IndexError:
                p_chs.append('#')
        str_p = "".join(p_chs)
        mapping.append(str_g + "~" + str_p)

    return mapping


def _dic_add(value, dic):
    if value not in dic:
        dic[value] = 1
    else:
        dic[value] += 1


def visualize_attention(sess, word, input_tensor, input_phon_tensor, output_phon_tensor, att_mats_encdec, encoder):
    word = [word]

    batch_input_tokenized = np.stack([_encode(input, encoder).squeeze(0) for input in word], 0)

    batch_phon_tokenized = sess.run(output_phon_tensor, feed_dict={input_tensor: batch_input_tokenized})

    batch_phon = [_decode(np.squeeze(phon_tokenized), encoder) for phon_tokenized in batch_phon_tokenized]

    encdec_att_mats = sess.run(att_mats_encdec,
                               feed_dict={input_tensor: batch_input_tokenized,
                                          input_phon_tensor: np.reshape(
                                              batch_phon_tokenized,
                                              [len(word), -1, 1, 1])})
    for i, encdec_att_mat in enumerate(encdec_att_mats):
        encdec_sum_all_layers = np.array(encdec_att_mat)[:len(batch_phon[0]), :len(word[0])]
        _plot_attention_matrix(word[0], batch_phon[0], encdec_sum_all_layers, 'Enc Dec Att L{}'.format(i + 1))
    _plot_attention_matrix(word[0], batch_phon[0],
                           np.sum(np.stack(encdec_att_mats), axis=0)[:len(batch_phon[0]), :len(word[0])],
                           'Enc Dec Att SUM')


def _plot_attention_matrix(inp_text, out_text, sum_all_layers, name):
    from matplotlib import pyplot as plt
    source_len = len(inp_text)
    prediction_len = len(out_text)

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(
        X=sum_all_layers,
        interpolation="nearest",
        cmap=plt.cm.Blues)
    plt.xticks(np.arange(source_len), inp_text, rotation=45)
    plt.yticks(np.arange(prediction_len), out_text, rotation=-45)
    fig.tight_layout()
    plt.show()


def load_model(model_dir, sess):
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
                    tempList.append(((int(lesson["LESSON"])), "".join(word), "".join(pred), ".".join(copy),
                                     len(word), len(pred)))

                    wordGp.remove((word, pred, gpMatch, copy))
        for word, pred, gpMatch, copy in wordGp[:]:
            tempList.append((999, "".join(word), "".join(pred), ".".join(copy), len(word), len(pred)))

        wordList = pd.DataFrame()
        wordList = wordList.append(tempList, ignore_index=True)
        wordList.columns = [["LESSON", "SPELLING", "PHONOLOGY", "GPMATCH", "N LETTERS", "N PHONEMES"]]
        return wordList
    else:
        tempList = []
        for word, pred, gpMatch, copy in wordGp[:]:
            tempList.append(("".join(word), " ".join(pred), ".".join(copy), len(word), len(pred)))
        wordList = pd.DataFrame()
        wordList = wordList.append(tempList, ignore_index=True)
        wordList.columns = [["SPELLING", "PHONOLOGY", "GPMATCH", "N LETTERS", "N PHONEMES"]]
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


def evaluate_gpa(sess, corpus, input_tensor, input_phon_tensor, output_phon_tensor, att_mats_list, encoder):
    batch_size = 128  # conservative batch size to dodge out of memory issues
    wordList = list(corpus.keys())
    wordCount = len(wordList)

    gp_results = []

    n_batch = wordCount // batch_size
    for idx_batch in tqdm(range(n_batch + 1), "Phon Translation"):
        try:
            batch = wordList[idx_batch * batch_size:(idx_batch + 1) * batch_size]
        except:
            batch = wordList[n_batch * batch_size:]

        _, _, batch_gp_results = g2p_mapping_batch(sess, batch, input_tensor, input_phon_tensor,
                                                   output_phon_tensor, att_mats_list, encoder)
        gp_results.extend([[results] for results in batch_gp_results])

    rates = error_rates(corpus, gp_results)
    print("WER : {:.4%} ; PER : {:.4%}".format(rates[0], rates[1]))


def stats(sess, wordList, phon, input_tensor, input_phon_tensor, output_phon_tensor, att_mats_list, encoder,
          weights, freq=None):
    if freq is not None:
        assert len(weights) == 4

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

    if " " in phon[0] and " " in phon[len(phon) - 1]:
        phon_results = [" ".join(phon) for phon in phon_results]
    else:
        phon_results = ["".join(phon) for phon in phon_results]
    gp_results = np.delete(np.array(gp_results), np.where(np.array(phon) != np.array(phon_results)))

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

    max_indiv_gpCount = max(gpCountDic.values())
    gpCount = sum(gpCountDic.values())
    tupList = []
    for i in range(len(gpList)):
        gp = gpList[i]
        g, p = gp.split("~")
        tup = (gp, gpCountDic[gp] / gpCount, gpCountDic[gp] / max_indiv_gpCount,
               gpCountDic[gp] / gCountDic[g], gpCountDic[gp] / pCountDic[p])
        tupList.append(tup)

    df = pd.DataFrame()
    df = df.append(tupList, ignore_index=True)
    df.columns = [["GP", "GP FREQ IN DATASET", "GP FREQ SCALED", "G CONSISTENCY", "P CONSISTENCY"]]

    if freq is None:
        scores = np.dot(weights, np.transpose(df[["GP FREQ SCALED", "G CONSISTENCY", "P CONSISTENCY"]]))
    else:
        scores = np.dot(weights,
                        np.concatenate(np.transpose(df[["GP FREQ SCALED", "G CONSISTENCY", "P CONSISTENCY"]]),
                                       freq))
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
    print(gd, pred)
    return 1


def error_rates(corpus, phon_results):
    wordList = list(corpus.keys())
    perS = 0
    werS = 0
    for i, word in tqdm(enumerate(wordList)):
        werS += wer(corpus[word], phon_results[i])
        perS += per(corpus[word], phon_results[i])
    return werS / len(wordList), perS / len(wordList)
