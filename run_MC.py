import torch
import numpy as np
import jsonlines
import json
import numpy as np
import math
import pprint
import os
import matplotlib.pyplot as plt
import logging
import argparse
import matplotlib
from transformers import RobertaTokenizer, RobertaForMultipleChoice, RobertaConfig, RobertaModel, BertTokenizer, PYTORCH_PRETRAINED_BERT_CACHE, BertForMultipleChoice, BertConfig
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import List
from torch import nn
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {

    'bert': (BertConfig, BertForMultipleChoice, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer)
}

def write_items(items, output_file):
    with open(output_file, 'w') as f:
        for item in items:
            f.write(str(item) + "\n")
    f.close()
    
def read_lines(input_file):
    lines = []
    with open(input_file, "rb") as f:
        for l in f:
            lines.append(l.decode().strip())
    return lines


def read_jsonl_lines(input_file):
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


class MultipleChoiceFeatures(object):
    def __init__(self,
                 example_id,
                 option_features,
                 label=None):
        self.example_id = example_id
        self.option_features = self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in option_features
        ]
        if label is not None:
            self.label = int(label) - 1
        else:
            self.label = None


class AnliExample(object):
    def __init__(self,
                 example_id,
                 beginning: str,
                 middle_options: list,
                 ending: str,
                 label=None):
        self.example_id = example_id
        self.beginning = beginning
        self.ending = ending
        self.middle_options = middle_options
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        lines = [
            "example_id:\t{}".format(self.example_id),
            "obs1:\t{}".format(self.beginning)
        ]
        for idx, option in enumerate(self.middle_options):
            lines.append("hyp{}:\t{}".format(idx, option))

        lines.append("obs2:\t{}".format(self.ending))

        if self.label is not None:
            lines.append("label:\t{}".format(self.label))
        return ", ".join(lines)

    def to_json(self):
        return {
            "story_id": self.example_id,
            "obs1": self.beginning,
            "obs2": self.ending,
            "hyp1": self.middle_options[0],
            "hyp2": self.middle_options[1],
            "label": self.label
        }

    def be_m_format(self):
        # (O1,O2) + H
        return [{
            "segment1": ' '.join([self.beginning, self.ending]),
            "segment2": hype
        } for hype in self.middle_options]
    
    def bm_e_format(self):
        # (O1,H) + O2
        return [{
            "segment1": ' '.join([self.beginning, hype]),
            "segment2": self.ending
        } for hype in self.middle_options]
    
    def b_me_format(self):
        # O1 + (H,O2)
        return [{
            "segment1": self.beginning,
            "segment2": ' '.join([hype, self.ending])
        } for hype in self.middle_options]
    
    def b_m_e_format(self):
        # (O1) + (H) + (O2)
        return [{
            "segment1": self.beginning,
            "segment2": hype,
            "segment3": self.ending
        } for hype in self.middle_options]

    def get_input_combination(self, _format):
        funcs = {
            "be_m" : self.be_m_format,
            "bm_e" : self.bm_e_format,
            "b_me" : self.b_me_format,
            "b_m_e" : self.b_m_e_format
        }
        return funcs[_format]()


class AnliProcessor():
    """Processor for the ANLI data set."""

    @classmethod
    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        records = []
        with open(input_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                records.append(obj)
        return records


    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.jsonl")))
        return self.get_examples_from_file(
            os.path.join(data_dir, "train.jsonl"),
            os.path.join(data_dir, "train-labels.lst"),
            "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "dev.jsonl")))
        return self.get_examples_from_file(
            os.path.join(data_dir, "dev.jsonl"),
            os.path.join(data_dir, "dev-labels.lst"),
            "train"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "test.jsonl")))
        return self.get_examples_from_file(
            os.path.join(data_dir, "test.jsonl"),
            os.path.join(data_dir, "test-labels.lst"),
            "train"
        )

    def get_examples_from_file(self, input_file, labels_file=None, split="predict"):
        if labels_file is not None:
            return self._create_examples(
                self._read_jsonl(input_file),
                read_lines(labels_file),
                split
            )
        else:
            return self._create_examples(
                self._read_jsonl(input_file)
            )

    def get_labels(self):
        """See base class."""
        return ["1", "2"]

    def _create_examples(self, records, labels=None, set_type="predict"):
        """Creates examples for the training and dev sets."""
        examples = []

        if labels is None:
            labels = [None] * len(records)

        for (i, (record, label)) in enumerate(zip(records, labels)):
            guid = "%s" % (record['story_id'])

            beginning = record['obs1']
            ending = record['obs2']

            option1 = record['hyp1']
            option2 = record['hyp2']

            examples.append(
                AnliExample(example_id=guid,
                            beginning=beginning,
                            middle_options=[option1, option2],
                            ending=ending,
                            label=label
                            )
            )
        return examples

    def label_field(self):
        return "label"


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_multiple_choice_examples_to_features(examples,
                                                 tokenizer,
                                                 max_seq_length,
                                                 is_training,
                                                 _format,
                                                 verbose = False):
    features = []
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    for idx, example in enumerate(examples):
        option_features = []
        for option in example.get_input_combination(_format):
            context_tokens = tokenizer.tokenize(option['segment1'])
            context_ids = tokenizer.convert_tokens_to_ids(context_tokens)
            option_tokens = tokenizer.tokenize(option["segment2"])
            option_ids = tokenizer.convert_tokens_to_ids(option_tokens)
            _truncate_seq_pair(context_ids, option_ids, max_seq_length - 3)
            token_ids = [cls_id] + context_ids + [sep_id] + option_ids + [sep_id]
            segment_ids = [0] * (len(context_tokens) + 2) + [1] * (len(option_tokens) + 1)
          
            input_ids = token_ids
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            option_features.append((token_ids, input_ids, input_mask, segment_ids))

        label = example.label

        if idx < 5 and verbose:
            logger.info("*** Example ***")
            logger.info(f"example_id: {example.example_id}")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(
                    option_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"token_ids: {' '.join(map(str, token_ids))}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            if is_training:
                logger.info(f"label: {label}")

        features.append(
            MultipleChoiceFeatures(
                example_id=example.example_id,
                option_features=option_features,
                label=label
            )
        )

    return features


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def mc_examples_to_data_loader(examples,
                               tokenizer,
                               max_seq_length,
                               is_train,
                               batch_size,
                               _format,
                               is_predict = False,
                               verbose = False):
    features = convert_multiple_choice_examples_to_features(
        examples, tokenizer, max_seq_length, is_train, _format, verbose
    )
    if verbose:
        logger.info("  Num examples = %d", len(examples))

    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)

    if not is_predict:
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    if is_train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def _model_name(dir_name):
    return os.path.join(dir_name, "pytorch_model.bin")


def _compute_softmax(scores):
    """
    Compute softmax probability over raw logits. 
    Apply the Max Trick when Computing Softmax.
    """
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def num_correct(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def train(data_dir, output_dir, data_processor, model_name_or_path, lr, batch_size, epochs,
          model_type, max_seq_length, warmup_proportion, _format, debug=False, tune_bert=True, gpu_id=0, tb_dir=None,
          debug_samples=20, training_data_fraction=1.0, config_name = None, do_lower_case = False):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(output_dir))
    os.makedirs(output_dir)

    writer = None
    if tb_dir is not None:
        writer = SummaryWriter(tb_dir)

    train_examples = data_processor.get_train_examples(data_dir)

    if training_data_fraction < 1.0:
        num_train_examples = int(len(train_examples) * training_data_fraction)
        train_examples = random.sample(train_examples, num_train_examples)

    if debug:
        logging.info("*****[DEBUG MODE]*****")
        train_examples = train_examples[:debug_samples]
    num_train_steps = int(
        len(train_examples) / batch_size * epochs
    )

    # Pretrained Model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

    config = config_class.from_pretrained(
            config_name if config_name else model_name_or_path,
            num_labels=len(data_processor.get_labels()),
        )
    tokenizer = tokenizer_class.from_pretrained(
            model_name_or_path,
            do_lower_case=do_lower_case,
        )
    model = model_class.from_pretrained(
        model_name_or_path,
        config=config
    )
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    if writer:
        params_to_log_on_tb = [(k, v) for k, v in model.named_parameters() if
                               not k.startswith("bert")]

    t_total = num_train_steps

    train_dataloader = mc_examples_to_data_loader(train_examples, tokenizer, max_seq_length, True,
                                                  batch_size, _format, verbose=True)

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = math.floor(warmup_proportion * t_total), num_training_steps = t_total)

    global_step = 0

    logging.info("\n\n\n\n****** TRAINABLE PARAMETERS = {} ******** \n\n\n\n"
                 .format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    for epoch_num in trange(int(epochs), desc="Epoch"):

        model.train()

        assert model.training

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        batch_tqdm = tqdm(train_dataloader)

        current_correct = 0

        for step, batch in enumerate(batch_tqdm):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            if model_type == 'roberta':
                model_output = model(input_ids = input_ids, attention_mask = input_mask, labels = label_ids)
            else:
                model_output = model(input_ids = input_ids, token_type_ids = segment_ids, attention_mask = input_mask, labels = label_ids)
            loss = model_output[0]
            logits = model_output[1]

            current_correct += num_correct(logits.detach().cpu().numpy(),
                                           label_ids.to('cpu').numpy())

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            loss.backward()

            if (step + 1) % 1 == 0: 
                # modify learning rate with special warm up BERT uses
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if writer:
                    writer.add_scalar("loss", tr_loss / nb_tr_steps, global_step)
                    lrs = scheduler.get_lr()
                    writer.add_scalar("lr_pg_1", lrs[0], global_step)
                    writer.add_scalar("lr_pg_2", lrs[1], global_step)
                    for n, p in params_to_log_on_tb:
                        writer.add_histogram(n, p.clone().cpu().data.numpy(), global_step)
                    writer.add_histogram("model_logits", logits.clone().cpu().data.numpy(),
                                         global_step)

            batch_tqdm.set_description(
                "Loss: {}; Iteration".format(round(tr_loss / nb_tr_steps, 3)))

        tr_acc = current_correct / nb_tr_examples

        result = {}
        if os.path.exists(output_dir+'train_metrics.json'):
            with open(output_dir+'train_metrics.json') as f:
                existing_results = json.loads(f.read())
            f.close()
            result.update(existing_results)

        result.update(
            {
                'train_loss': tr_loss / nb_tr_steps,
                'train_accuracy': tr_acc,
            }
        )

        with open(output_dir+'train_metrics.json', "w") as wri:
            wri.write(json.dumps(result))

        # Call evaluate at the end of each epoch
        result = evaluate(data_dir=data_dir,
                          output_dir=output_dir,
                          data_processor=data_processor,
                          model_name_or_path=model_name_or_path,
                          finetuning_model=finetuning_model,
                          max_seq_length=max_seq_length,
                          batch_size=batch_size,
                          _format = _format, 
                          debug=debug,
                          gpu_id=gpu_id,
                          model=model,
                          tokenizer=tokenizer,
                          verbose=False,
                          debug_samples=debug_samples,
                          do_lower_case = do_lower_case,
                          )
        logging.info("****** EPOCH {} ******\n\n\n".format(epoch_num))
        logging.info("Training Loss: {}".format(round(tr_loss / nb_tr_steps, 3)))
        logging.info("Training Accuracy: {}".format(round(tr_acc, 3)))
        logging.info("Validation Loss : {}".format(round(result['dev_eval_loss'], 3)))
        logging.info("Validation Accuracy : {}".format(round(result['dev_eval_accuracy'], 3)))
        logging.info("******")

        if writer:
            writer.add_scalar("dev_val_loss", result['dev_eval_loss'], global_step)
            writer.add_scalar("dev_val_accuracy", result['dev_eval_accuracy'], global_step)
            writer.add_scalar("dev_accuracy", tr_acc, global_step)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = _model_name(output_dir)
    torch.save(model_to_save.state_dict(), output_model_file)
    logging.info("Training Done. Saved model to: {}".format(output_model_file))
    return output_model_file


def evaluate(data_dir, output_dir, data_processor, model_name_or_path, model_type, max_seq_length,
             batch_size, _format, debug=False, gpu_id=0, model=None, tokenizer=None, verbose=False, debug_samples=20,
             eval_split="dev", config_name=None, metrics_out_file="metrics.json", do_lower_case = False):
    if debug:
        logging.info("*****[DEBUG MODE]*****")
        eval_examples = data_processor.get_train_examples(data_dir)[:debug_samples]
    else:
        if eval_split == "dev":
            eval_examples = data_processor.get_dev_examples(data_dir)
        elif eval_split == "test":
            eval_examples = data_processor.get_test_examples(data_dir)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    if tokenizer is None:
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)

    eval_dataloader = mc_examples_to_data_loader(examples=eval_examples,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=max_seq_length,
                                                 is_train=False,
                                                 batch_size=batch_size,
                                                 _format = _format,
                                                 verbose=verbose
                                                 )

    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    # Load a trained model
    if model is None:
        config = config_class.from_pretrained(
            config_name if config_name else model_name_or_path,
            num_labels=len(data_processor.get_labels()),
        )
        model = model_class.from_pretrained(
        model_name_or_path,
        config=config
        )
        model.to(device)

    model.eval()

    assert not model.training

    eval_loss, eval_correct = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    eval_predictions = []
    eval_logits = []
    eval_pred_probs = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            if model_type == 'roberta':
                model_output = model(input_ids = input_ids, attention_mask = input_mask, labels = label_ids)
            else:
                model_output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            tmp_eval_loss = model_output[0]
            logits = model_output[1]

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_correct = num_correct(logits, label_ids)

        eval_predictions.extend(np.argmax(logits, axis=1).tolist())
        eval_logits.extend(logits.tolist())
        eval_pred_probs.extend([_compute_softmax(list(l)) for l in logits])

        eval_loss += tmp_eval_loss.item()  # No need to compute mean again. CrossEntropyLoss does that by default.
        nb_eval_steps += 1

        eval_correct += tmp_eval_correct
        nb_eval_examples += input_ids.size(0)

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_correct / nb_eval_examples

    result = {}
    if os.path.exists(metrics_out_file):
        with open(metrics_out_file) as f:
            existing_results = json.loads(f.read())
        f.close()
        result.update(existing_results)

    result.update(
        {
            eval_split + '_eval_loss': eval_loss,
            eval_split + '_eval_accuracy': eval_accuracy,
        }
    )

    with open(metrics_out_file, "w") as writer:
        writer.write(json.dumps(result))

    if verbose:
        logger.info("***** Eval results *****")
        logging.info(json.dumps(result))

    output_file = os.path.join(os.path.dirname(output_dir),
                               eval_split + "_output_predictions.jsonl")

    predictions = []
    for record, pred, logits, probs in zip(eval_examples, eval_predictions, eval_logits,
                                           eval_pred_probs):
        r_json = record.to_json()
        r_json['prediction'] = data_processor.get_labels()[pred]
        r_json['logits'] = logits
        r_json['probs'] = probs
        predictions.append(r_json)
    write_items([json.dumps(r) for r in predictions], output_file)

    return result


def predict(pred_input_file,
            pred_output_file,
            model_dir,
            data_processor,
            model_name_or_path,
            max_seq_length,
            batch_size,
            gpu_id,
            _format,
            verbose,
            model_type,
            config_name=None,
            do_lower_case = False):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
    pred_examples = data_processor.get_examples_from_file(pred_input_file)

    pred_dataloader = mc_examples_to_data_loader(examples=pred_examples,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=max_seq_length,
                                                 is_train=False,
                                                 is_predict=True,
                                                 batch_size=batch_size,
                                                 _format = _format,
                                                 verbose=verbose
                                                 )

    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    # Load a trained model that you have fine-tuned
    if torch.cuda.is_available():
        model_state_dict = torch.load(_model_name(model_dir))
    else:
        model_state_dict = torch.load(_model_name(model_dir), map_location='cpu')

    # Pretrained Model
    config = config_class.from_pretrained(
            config_name if config_name else model_name_or_path,
            num_labels=len(data_processor.get_labels()),
        )
    model = model_class.from_pretrained(
    model_name_or_path,
    config=config
    )
    model.to(device)


    model.eval()

    assert not model.training

    predictions = []

    for input_ids, input_mask, segment_ids in tqdm(pred_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            if model_type == 'roberta':
                model_output = model(input_ids = input_ids, attention_mask = input_mask, labels = label_ids)
            else:
                model_output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            logits = model_output[0]

        logits = logits.detach().cpu().numpy()

        predictions.extend(np.argmax(logits, axis=1).tolist())

    write_items([idx + 1 for idx in predictions], pred_output_file)


def main(args):
    seed = args.seed
    model_name_or_path = args.model_name_or_path
    do_lower_case = args.do_lower_case
    data_dir = args.data_dir
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    max_seq_length = args.max_seq_length
    warmup_proportion = args.warmup_proportion
    mode = args.mode
    model_type = args.finetuning_model.lower()
    debug = args.debug
    tune_bert = not args.no_tune_bert
    gpu_id = args.gpu_id
    debug_samples = args.debug_samples
    run_on_test = args.run_on_test
    training_data_fraction = args.training_data_fraction
    run_on_dev = True
    _format = args._format

    exp_dir = '%s_%s_E%d_B%d_LR%s_%s' % (model_name_or_path, model_type, epochs, batch_size, lr, _format)
    output_dir = os.path.join(args.output_dir, exp_dir)
    print(output_dir)
    tb_dir = output_dir + '/tb/'
    print(tb_dir)
    metrics_out_file = output_dir + '/metrics.json'
    print(metrics_out_file)


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if mode is None or mode == "train":
        train(data_dir=data_dir,
              output_dir=output_dir,
              data_processor=AnliProcessor(),
              model_name_or_path=model_name_or_path,
              lr=lr,
              batch_size=batch_size,
              epochs=epochs,
              model_type=model_type,
              max_seq_length=max_seq_length,
              warmup_proportion=warmup_proportion,
              debug=debug,
              tune_bert=tune_bert,
              gpu_id=gpu_id,
              tb_dir=tb_dir,
              debug_samples=debug_samples,
              training_data_fraction=training_data_fraction,
              do_lower_case = do_lower_case,
              _format = _format
              )
    if mode is None or mode == "eval":
        if run_on_dev:
            evaluate(
                data_dir=data_dir,
                output_dir=output_dir,
                data_processor=AnliProcessor(),
                model_name_or_path=model_name_or_path,
                model_type=model_type,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
                debug=debug,
                gpu_id=gpu_id,
                verbose=True,
                debug_samples=debug_samples,
                eval_split="dev",
                metrics_out_file=metrics_out_file,
                do_lower_case = do_lower_case,
                _format = _format
            )

        if run_on_test:
            logger.info("*******")
            logger.info("!!!!!!! ----- RUNNING ON TEST ----- !!!!!")
            logger.info("*******")
            evaluate(
                data_dir=data_dir,
                output_dir=output_dir,
                data_processor=AnliProcessor(),
                model_name_or_path=model_name_or_path,
                model_type=model_type,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
                debug=debug,
                gpu_id=gpu_id,
                verbose=True,
                debug_samples=debug_samples,
                eval_split="test",
                metrics_out_file=metrics_out_file,
                do_lower_case = do_lower_case,
                _format = _format
            )

    if mode == "predict":
        assert args.predict_input_file is not None and args.predict_output_file is not None

        predict(
            pred_input_file=args.predict_input_file,
            pred_output_file=args.predict_output_file,
            model_dir=output_dir,
            data_processor=AnliProcessor(),
            model_name_or_path=model_name_or_path,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            gpu_id=gpu_id,
            verbose=False,
            model_type=model_type,
            do_lower_case = do_lower_case,
            _format = _format
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Finetune Roberta model and save')

    # Required Parameters
    parser.add_argument('--model_name_or_path',
                        type=str,
                        help="BERT and RoBERTa pre-trained model selected for finetuned",
                        default=None)
    
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--finetuning_model', type=str, default='BERTForMultipleChoice')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--eval_split', type=str, default="dev")
    parser.add_argument('--run_on_test', action='store_true')

    parser.add_argument('--input_file', action='store_true')
    parser.add_argument('--predict_input_file', default=None)
    parser.add_argument('--predict_output_file', default=None)
   

    # Hyperparams
    parser.add_argument('--lr', type=float, help="Learning rate", default=1e-5)
    parser.add_argument('--batch_size', type=int, help="Batch size", default=4)
    parser.add_argument('--epochs', type=int, help="Num epochs", default=3)
    parser.add_argument('--training_data_fraction', type=float, default=1.0)
    parser.add_argument('--_format', type=str, help="pre-trained model input pattern", default='be_m')

    # Other parameters
    parser.add_argument('--data_dir', type=str, help='Location of data', default='data/')
    parser.add_argument('--output_dir',
                        type=str,
                        help="Output directory to save model",
                        default='models/')
    # parser.add_argument('--metrics_out_file', default="metrics.json")
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--warmup_proportion',
                        type=float,
                        default=0.2,
                        help="Portion of training to perform warmup")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_samples', default=20, type=int)
    parser.add_argument('--no_tune_bert', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    # parser.add_argument('--tb_dir', type=str, default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)