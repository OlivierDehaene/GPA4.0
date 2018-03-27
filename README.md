# Word2Phonetic

[Word2Phonetic](https://github.com/OlivierDehaene/word2phonetic), or
[w2p](https://github.com/OlivierDehaene/word2phonetic) is a submodule developed for the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) library. 
It provides an easy to use grapheme-phoneme alignement tool for any given language.
[Find out more](http://www.unicog.org/).


### Quick Start

```
# Clone the repository
git clone https://github.com/OlivierDehaene/word2phonetic

# Install the w2p package and its dependencies
cd word2phonetic
pip install -e .

# Train
DATA_DIR=$HOME/w2p_data
TRAIN_DIR=$HOME/w2p_checkpoints

w2p-train # a simple t2t-trainer wrapper with w2p default settings
  --data_dir=$DATA_DIR \
  --output_dir=$TRAIN_DIR

# Decode 

DECODE_FILE=./decode_this.txt
echo "Hello" >> $DECODE_FILE
echo "world" >> $DECODE_FILE

w2p-decoder 
  --model_dir=$TRAIN_DIR \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=phonoly_gpmatch.csv
  
# See the phonology translation and the grapheme-phoneme alignement
cat translation.en
```

Go to the [Wiki] page for more informations.
