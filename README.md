# Abductive Natural Language Inference with Pre-trained Deep Language Models

# Tianxiang Wang, Zhe Fan (Group 6)


## Usage

### Set up environment

L2R2 is tested on Python 3.7, PyTorch 1.0.1, and transformers==2.10.0


### Prepare data

[αNLI](https://leaderboard.allenai.org/anli/submissions/get-started)
```shell script
$ wget https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip
$ unzip -d alphanli alphanli-train-dev.zip
```
### Statistics

The statistics of the corpora is give by `stats.py`


### Baseline

We implemented perceptron classifier and decision tree with handcrafted features


### Fine-tuning pre-trained models

We fine-tuned the pre-trained models on 1 Titan X GPU. 

The available `model` for fine-tuning could selected in:
- BERT base uncased Multiplce Choice
- BERT large Multiplce Choice
- RoBERTa base uncased Multiplce Choice
- RoBERTa large uncased Multiplce Choice

### L2R format

Traininf instances could be formulated into format which is suitable for learning to rank
`process.py`

### L2R losses

We have implemented two learn to rank losses in `losses.py`
- RankNet: pair-wise logistic loss
- ListNet: list-wise K-L Divegence
