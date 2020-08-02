# coding=utf-8
import argparse
import json
import numpy as np

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

def _key(r):
    return r['obs1'] + '||' + r['obs2']


def correct_hypothese(r):
    return r['hyp' + r['label']]


def incorrect_hypothese(r):
    if r['label'] == "1":
        return r['hyp2']
    else:
        return r['hyp1']


def mean_word_lens(lst):
    return round(np.mean([len(s.split()) for s in lst]),2)


def main(args):
    input_file = args.input_file
    labels_file = args.label_file

    stories = read_jsonl_lines(input_file)
    labels = read_lines(labels_file)

    all_begins = []
    all_endings = []
    num_duplicate = 0

    stories_by_key = {}
    for s, label in zip(stories, labels):
        s['label'] = label

        if s['hyp1'] == s['hyp2']:
            num_duplicate += 1

        key = _key(s)
        if key not in stories_by_key:
            stories_by_key[key] = []
        stories_by_key[key].append(s)

        all_begins.append(s['obs1'])
        all_endings.append(s['obs2'])

    num_correct_hypotheses_per_story = []
    num_incorrect_hypotheses_per_story = []
    all_correct_hypotheses = []
    all_incorrect_hypotheses = []

    all_begins = list(set(all_begins))
    all_endings = list(set(all_endings))

    for k, stories in stories_by_key.items():
        num_correct_hypotheses_per_story.append(len(set([correct_hypothese(r) for r in stories])))
        num_incorrect_hypotheses_per_story.append(len(set([incorrect_hypothese(r) for r in stories])))

        all_correct_hypotheses.extend(list(set([correct_hypothese(r) for r in stories])))
        all_incorrect_hypotheses.extend(list(set([incorrect_hypothese(r) for r in stories])))

    print("No. of train stories: {}".format(len(stories_by_key)))
    print("Mean of no. of correct hypotheses = {}".format(round(np.mean(num_correct_hypotheses_per_story), 2)))
    print("Mean of no. of incorrect hypotheses = {}".format(round(np.mean(num_incorrect_hypotheses_per_story), 2)))

    print("Duplicate hypotheses samples = {}".format(num_duplicate))

    print("Mean of no. of words in correct hypotheses = {}".format(mean_word_lens(all_correct_hypotheses)))
    print("Mean of no. of words in incorrect hypotheses = {}".format(mean_word_lens(all_incorrect_hypotheses)))

    print("No. correct hypotheses = {}".format(len(all_correct_hypotheses)))
    print("No. incorrect hypotheses = {}".format(len(all_incorrect_hypotheses)))

    print("Mean of no. of words in Obs1s = {}".format(mean_word_lens(all_begins)))
    print("Mean of no. of words in Obs2s = {}".format(mean_word_lens(all_endings)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to compute corpus statistics')

    # Required Parameters
    parser.add_argument('--input_file', type=str, help='Location of data', default=None)
    parser.add_argument('--label_file', type=str, help='Location of data', default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)