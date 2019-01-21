# GPA 4.0

[GPA4.0](https://github.com/OlivierDehaene/GPA4.0), or
[gpa](https://github.com/OlivierDehaene/GPA4.0) is a submodule developed for the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) library. It does Grapheme-to-Phoneme (G2P) conversion and Grapheme-Phoneme-Alignement using the Transformer model. Transformer is a sequence-to-sequence model that was successfully applied to numerous tasks, including Neural Machine Translation [[1]](https://arxiv.org/pdf/1706.03762.pdf). G2P is a task that was solved in the past using LSTM RNN based architectures [[2]](https://arxiv.org/pdf/1506.00196.pdf) [[3]](https://arxiv.org/pdf/1610.06540.pdf). We build on these results and propose a new G2P and semi-supervised GPAlignement tool.
[Find out more](https://github.com/OlivierDehaene/GPA4.0/wiki).

### Results

|Data set|WER|PER|
|---|---|---|
|CMU|20.8667%|4.5933%|
|FR|3.1831%|0.8889%|
|BR|2.7672%|0.5485%|
|ES|0.3784%|0.0744%|

### Quick Start

```
# Clone the repository
git clone https://github.com/OlivierDehaene/GPA4.0

# Install the gpa package and its dependencies
cd GPA4.0
pip install -e .

# Train
DATA_DIR=$HOME/gpa_data
TRAIN_DIR=$HOME/gpa_checkpoints

gpa-train # a simple t2t-trainer wrapper with gpa default settings
  --data_dir=$DATA_DIR \
  --output_dir=$TRAIN_DIR

# Decode 

DECODE_FILE=./decode_this.txt
echo "Hello" >> $DECODE_FILE
echo "world" >> $DECODE_FILE

gpa-decoder 
  --model_dir=$TRAIN_DIR \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=phonology_gpmatch.csv \
  --data_dir=$DATA_DIR
  
# See the phonology translation and the grapheme-phoneme alignement
cat phonology_gpmatch.csv
```
