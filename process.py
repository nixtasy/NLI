import collections
import logging
import json
logger = logging.getLogger(__name__)
from tqdm import tqdm, tqdm_notebook
from transformers import BertTokenizer
import torch
import os

# process the data into l2r form

class StoryExample(object):

    def __init__(self, example_id, obs1, obs2, hypes, labels=None):

        self.example_id = example_id
        self.obs1 = obs1
        self.obs2 = obs2
        self.hypes = hypes
        self.hyp2idx = dict([(hyp, i) for i, hyp in enumerate(hypes)])
        self.labels = labels

    @classmethod
    def from_dict(cls, dic):
        return cls(**dic)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return json.dumps(self.__dict__, ensure_ascii=False)

class AlphaNliProcessor(object):

    def __init__(self, data_dir, tokenizer):
        """

        Args:
            data_dir (str):
            tokenizer (PreTrainedTokenizer):
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
    
    def get_examples(self, mode='train', force_preprocess=False):
        """

        Args:
            mode:
            force_preprocess:

        Returns:
            list[StoryExample]:
        """
        logger.info("***** Loading %s examples *****", mode)
        cache_filename = '%s.examples' % mode
        cache_file_path = os.path.join(self.data_dir, cache_filename)
        if not os.path.exists(cache_file_path) or force_preprocess:
            self._preprocess(os.path.join(self.data_dir, cache_filename.replace('examples', 'jsonl')),
                             os.path.join(self.data_dir, '%s-labels.lst' % mode),
                             cache_file_path)
        return torch.load(cache_file_path)

    def _preprocess(self, sample_file, label_file, output_file):
            logger.info("***** Pre-processing %s *****" % sample_file)
            stories = collections.defaultdict(dict)
            num_duplicate = 0
            for sample in tqdm(self.get_samples(sample_file, label_file)):
                label = sample['label']
                # skip over the samples which contain duplicated hypes
                if sample['hyp1'] == sample['hyp2']:
                    num_duplicate += 1
                    continue
                if sample['story_id'] not in stories:
                    stories[sample['story_id']]['cnt'] = 1
                    stories[sample['story_id']]['obs1'] = sample['obs1']
                    stories[sample['story_id']]['obs2'] = sample['obs2']
                    stories[sample['story_id']]['hypes'] = collections.defaultdict(lambda: [0, 0])
                    stories[sample['story_id']]['hypes'][sample['hyp'+str(label)]][0] = 1
                    stories[sample['story_id']]['hypes'][sample['hyp1']][1] = 1
                    stories[sample['story_id']]['hypes'][sample['hyp2']][1] = 1
                else:
                    stories[sample['story_id']]['cnt'] += 1
                    assert stories[sample['story_id']]['obs1'] == sample['obs1']
                    assert stories[sample['story_id']]['obs2'] == sample['obs2']
                    stories[sample['story_id']]['hypes'][sample['hyp'+str(label)]][0] += 1
                    stories[sample['story_id']]['hypes'][sample['hyp1']][1] += 1
                    stories[sample['story_id']]['hypes'][sample['hyp2']][1] += 1
            logger.info('%d duplicate hypotheses samples:', num_duplicate)
            examples = []
            for _id, story in stories.items():
                examples.append(StoryExample(_id, story['obs1'], story['obs2'],
                                            list(story['hypes'].keys()),
                                            [(n_pos,n_sum) for n_pos, n_sum in story['hypes'].values()]))
            torch.save(examples, output_file)
            logger.info("***** Saved to %s *****" % output_file)
    
    @classmethod
    def read_json(cls, input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
        return data

    @classmethod
    def read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        records = []
        with open(input_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                records.append(obj)
        return records
    
    @classmethod
    def read_lst(cls, input_file):
        """Reads a lst file."""
        with open(input_file, "r") as f:
            lines = f.readlines()
        return lines

    @classmethod
    def get_samples(cls, sample_file, label_file):
        samples = []
        for sample, label in tqdm(zip(cls.read_jsonl(sample_file), cls.read_lst(label_file))):
            sample['label'] = int(label)
            samples.append(sample)
        return samples

# tokenizer = BertTokenizer.from_pretrained(
#             'bert-base-uncased',
#             do_lower_case = True,
#             cache_dir = None,
#         )
# processor = AlphaNliProcessor('./', tokenizer)
# processor.get_examples()