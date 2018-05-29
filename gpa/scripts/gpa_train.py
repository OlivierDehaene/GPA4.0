"""
Wrapper for t2t-trainer with w2p settings selected as defaults
"""
from tensor2tensor.bin import t2t_trainer

import tensorflow as tf
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

flags = tf.flags
FLAGS = flags.FLAGS

def main(argv):
    FLAGS.t2t_usr_dir = os.path.join(__location__, "../submodule")
    FLAGS.hparams_set = "g2p"
    FLAGS.model = "transformer"
    FLAGS.problem = "grapheme_to_phoneme"
    if FLAGS.train_steps == None:
        FLAGS.train_steps = 10000
    FLAGS.generate_data = True

    t2t_trainer.main(argv)


if __name__ == "__main__":
    tf.app.run()
