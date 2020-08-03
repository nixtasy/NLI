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

```shell script
$ python stats.py \
  --input_file data/train.jsonl\
  --label_file data/train-labels.lst
```

### Baseline

We implemented perceptron classifier and decision tree with handcrafted features


### Fine-tuning pre-trained multiple choice classfication models

We fine-tuned the pre-trained models on 1 Titan X GPU. 

The available `model` for fine-tuning could selected in:
- BERT base uncased Multiplce Choice
- BERT large Multiplce Choice
- RoBERTa base uncased Multiplce Choice
- RoBERTa large uncased Multiplce Choice

```shell script
$ python run_MC.py\
  --model_name_or_path bert-base-uncased\
  --do_lower_case\
  --batch_size 8\
  --lr 1e-5\
  --epochs 4\
  --finetuning_model bert\
  --max_seq_length 68\
  --seed 21004
```

```shell script
$ python run_MC.py\
  --model_name_or_path roberta-base\
  --batch_size 8\
  --lr 1e-5\
  --epochs 4\
  --finetuning_model roberta\
  --max_seq_length 68\
  --seed 21004
```


### L2R format

Traininf instances could be formulated into format which is suitable for learning to rank
`process.py`

### L2R losses

We have implemented two learn to rank losses in `losses.py`
- RankNet: pair-wise logistic loss
- ListNet: list-wise K-L Divegence
