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

from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

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

    for i in range(translate_model.hparams.num_hidden_layers):
        encdec_att = translate_model.attention_weights[
            '%sdecoder/layer_%i/encdec_attention%s' % (prefix, i, postfix)]
        encdec_atts.append(encdec_att)

    selected_heads = [0, 4,
                      5]  # in our experience, the first, the second last and the last layers are the most interpretable
    att_mats = [tf.squeeze(tf.reduce_sum(encdec_atts[head], axis=1)) for head in selected_heads]

    return att_mats


def build_model(hparams_set, model_name, data_dir, problem_name, beam_size=1):
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
        }, beam_size=beam_size)['outputs']

    return inputs, targets, samples, att_mats


def export_model(sess, input_tensor, output_phon_tensor, input_phon_tensor, att_mats_list, output_dir):
    input_get_phon = {'input': tf.saved_model.utils.build_tensor_info(input_tensor)}

    output_get_phon = {'phon': tf.saved_model.utils.build_tensor_info(output_phon_tensor)}

    signature_get_phon = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=input_get_phon,
        outputs=output_get_phon,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    input_get_att_mats = {'input': tf.saved_model.utils.build_tensor_info(input_tensor),
                          'phon': tf.saved_model.utils.build_tensor_info(input_phon_tensor)}

    output_get_att_mats = {'att_mat_inp_out_layer_0': tf.saved_model.utils.build_tensor_info(att_mats_list[0]),
                           'att_mat_inp_out_layer_4': tf.saved_model.utils.build_tensor_info(att_mats_list[1]),
                           'att_mat_inp_out_layer_5': tf.saved_model.utils.build_tensor_info(att_mats_list[2])}

    signature_get_att_mats = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=input_get_att_mats,
        outputs=output_get_att_mats,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    # Save out the SavedModel.
    builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'get_phon': signature_get_phon,
            'get_att_mats': signature_get_att_mats
        },
        legacy_init_op=legacy_init_op)
    builder.save()


def _load_model(model_dir, sess):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
        return True
    return False


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--problem_name', type=str, default="word_to_phonetic")
    parser.add_argument('--model_name', type=str, default="transformer")
    parser.add_argument('--hparams_set', type=str, default="w2p")
    parser.add_argument('--t2t_usr_dir', type=str, default=os.path.join(__location__, "../submodule"))
    args = parser.parse_args()

    usr_dir.import_usr_dir(args.t2t_usr_dir)
    inputs, targets, samples, att_mats = build_model(args.hparams_set, args.model_name, args.data_dir, args.problem_name, beam_size=5)

    sess = tf.Session()

    assert _load_model(args.checkpoint_dir, sess)

    export_model(sess=sess, input_tensor=inputs, output_phon_tensor=samples, input_phon_tensor=targets,
                 att_mats_list=att_mats, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
