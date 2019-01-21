"""
Wrapper for t2t-trainer with w2p settings selected as defaults
"""
import tensorflow as tf
import os

from tensor2tensor.bin import t2t_trainer

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

flags = tf.flags
FLAGS = flags.FLAGS


def main(argv):
    FLAGS.t2t_usr_dir = os.path.join(__location__, "../submodule")
    FLAGS.hparams_set = "g2p"
    FLAGS.model = "transformer"
    FLAGS.problem = "grapheme_to_phoneme"
    FLAGS.train_steps = 100000
    FLAGS.generate_data = True

    t2t_trainer.main(argv)


if __name__ == "__main__":
    tf.app.run()
