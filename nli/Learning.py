from KB import KB
from Utils import  Utils
from Perceptron import  Perceptron
import jsonlines
import math
import time
import random

"""
feature 1: for every tuple instance, "1" if a word from h, also occurs in both o1,o2; else 0.
feature 2: for every tuple instance pairs, "1" if one of h has more such words as shown in feature 1 than another; else 0
feature 3: for every tuple instance pairs, "1" if one of h reach smaller distance (calculated by : abs(len(o1)-len(h))+abs(len(o2)-len(h))) than another; else 0
feature 4: for every tuple instance pairs, "1" if one of h reach higher cosine similarities (calculated by : cosim(o1,h)) than another; else 0
feature 5: for every tuple instance pairs, "1" if one of h reach higher cosine similarities (calculated by : cosim(o2,h)) than another; else 0
"""

class Learning:

    def __init__(self):
        self.train = []
        self.dev  = []
        self.train_data_path = 'jsonl/train.jsonl'
        self.dev_data_path = 'jsonl/dev.jsonl'
        self.train_label_path = 'jsonl/train-labels.lst'
        self.dev_label_path = 'jsonl/dev-labels.lst'

    def ingest_data(self):
        """
        First load raw train and dev data,
        split hypothesises, so that each instance contains
        (o1,o2,h1/h2) and a label(1/0)

        """
        with jsonlines.open(self.train_data_path) as reader:
            train_data = [obj for obj in reader]

        with jsonlines.open(self.train_label_path) as reader:
            train_label = [obj for obj in reader]

        for X, Y in zip(train_data, train_label):
            self.train.append(KB(X['obs1'],X['obs2'],X['hyp1'],1 if Y == 1 else 0))
            self.train.append(KB(X['obs1'], X['obs2'], X['hyp2'], 1 if Y == 2 else 0))
        # for X, Y in zip(train_data, train_label):
        #     self.train.append(KB(X['obs1'],X['obs2'],X['hyp1'], X['hyp2'],Y))

        with jsonlines.open(self.dev_data_path) as reader:
            dev_data = [obj for obj in reader]

        with jsonlines.open(self.dev_label_path) as reader:
            dev_label = [obj for obj in reader]

        for X, Y in zip(dev_data, dev_label):
            self.dev.append(KB(X['obs1'],X['obs2'],X['hyp1'],1 if Y == 1 else 0))
            self.dev.append(KB(X['obs1'], X['obs2'], X['hyp2'], 1 if Y == 2 else 0))
        # for X, Y in zip(dev_data, dev_label):
        #     self.dev.append(KB(X['obs1'],X['obs2'],X['hyp1'], X['hyp2'],Y))


    def generate_cross_features(self, data='train'):
        if data == 'train':
            for t in range(len(self.train)):
                vocabulary = []
                if t % 2 == 0:
                    o1 = [token.lower() for token in self.train[t].o1.split()]
                    o2 = [token.lower() for token in self.train[t].o2.split()]
                    h1 = [token.lower() for token in self.train[t].h.split()]
                    h2 = [token.lower() for token in self.train[t + 1].h.split()]

                    o1[-1] = o1[-1][:-1]
                    o2[-1] = o2[-1][:-1]
                    h1[-1] = h1[-1][:-1]
                    h2[-1] = h2[-1][:-1]

                    if self.train[t].F[0] + self.train[t + 1].F[0] != 0:
                        overlap_score = [0] * 2
                        for h in h1:
                            if h in o1 and h in o2:
                                overlap_score[0] += 1
                        for h in h2:
                            if h in o1 and h in o2:
                                overlap_score[1] += 1
                        if overlap_score[0] == overlap_score[1]:
                            self.train[t].add_feature(1)
                            self.train[t + 1].add_feature(1)
                        elif overlap_score[0] > overlap_score[1]:
                            self.train[t].add_feature(1)
                            self.train[t + 1].add_feature(0)
                        else:
                            self.train[t].add_feature(0)
                            self.train[t + 1].add_feature(1)
                    else:
                        self.train[t].add_feature(0)
                        self.train[t + 1].add_feature(0)

                    len_diff_h1 = abs(len(o1) - len(h1)) + abs(len(o2) - len(h1))
                    len_diff_h2 = abs(len(o1) - len(h2)) + abs(len(o2) - len(h2))
                    if len_diff_h1 > len_diff_h2:
                        self.train[t].add_feature(0)
                        self.train[t + 1].add_feature(1)
                    elif len_diff_h1 < len_diff_h2:
                        self.train[t].add_feature(1)
                        self.train[t + 1].add_feature(0)
                    else:
                        self.train[t].add_feature(1)
                        self.train[t + 1].add_feature(1)

                    vocabulary.extend(o1)
                    vocabulary.extend(o2)
                    vocabulary.extend(h1)
                    vocabulary.extend(h2)
                    vocabulary = list(set(vocabulary))

                    v1 = [0] * len(vocabulary)
                    v2 = [0] * len(vocabulary)
                    vh1 = [0] * len(vocabulary)
                    vh2 = [0] * len(vocabulary)

                    for token in o1:
                        v1[vocabulary.index(token)] += 1
                    for token in o2:
                        v2[vocabulary.index(token)] += 1
                    for token in h1:
                        vh1[vocabulary.index(token)] += 1
                    for token in h2:
                        vh2[vocabulary.index(token)] += 1

                    v1_len = 0
                    v2_len = 0
                    vh1_len = 0
                    vh2_len = 0

                    for i, j, k, r in zip(v1, v2, vh1, vh2):
                        if i:
                            v1_len += i * i
                        if j:
                            v2_len += j * j
                        if k:
                            vh1_len += k * k
                        if r:
                            vh2_len += r * r

                    v1_len = math.sqrt(v1_len)
                    v2_len = math.sqrt(v2_len)
                    vh1_len = math.sqrt(vh1_len)
                    vh2_len = math.sqrt(vh2_len)

                    cos_o1_h1 = sum([i * j for (i, j) in zip(v1, vh1)]) / (v1_len * vh1_len)
                    cos_o1_h2 = sum([i * j for (i, j) in zip(v1, vh2)]) / (v1_len * vh2_len)

                    if cos_o1_h1 > cos_o1_h2:
                        self.train[t].add_feature(1)
                        self.train[t + 1].add_feature(0)
                    elif cos_o1_h1 == cos_o1_h2 and cos_o1_h1 != 0:
                        self.train[t].add_feature(1)
                        self.train[t + 1].add_feature(1)
                    elif cos_o1_h1 < cos_o1_h2:
                        self.train[t].add_feature(0)
                        self.train[t + 1].add_feature(1)
                    else:
                        self.train[t].add_feature(0)
                        self.train[t + 1].add_feature(0)

                    cos_o2_h1 = sum([i * j for (i, j) in zip(v2, vh1)]) / (v2_len * vh1_len)
                    cos_o2_h2 = sum([i * j for (i, j) in zip(v2, vh2)]) / (v2_len * vh2_len)

                    if cos_o2_h1 > cos_o2_h2:
                        self.train[t].add_feature(1)
                        self.train[t + 1].add_feature(0)
                    elif cos_o2_h1 == cos_o2_h2 and cos_o2_h1 != 0:
                        self.train[t].add_feature(1)
                        self.train[t + 1].add_feature(1)
                    elif cos_o2_h1 < cos_o2_h2:
                        self.train[t].add_feature(0)
                        self.train[t + 1].add_feature(1)
                    else:
                        self.train[t].add_feature(1)
                        self.train[t + 1].add_feature(1)
        else:
            for d in range(len(self.dev)):
                vocabulary = []
                if d % 2 == 0:
                    o1 = [token.lower() for token in self.dev[d].o1.split()]
                    o2 = [token.lower() for token in self.dev[d].o2.split()]
                    h1 = [token.lower() for token in self.dev[d].h.split()]
                    h2 = [token.lower() for token in self.dev[d + 1].h.split()]

                    o1[-1] = o1[-1][:-1]
                    o2[-1] = o2[-1][:-1]
                    h1[-1] = h1[-1][:-1]
                    h2[-1] = h2[-1][:-1]

                    if self.dev[d].F[0] + self.dev[d + 1].F[0] != 0:
                        overlap_score = [0] * 2
                        for h in h1:
                            if h in o1 and h in o2:
                                overlap_score[0] += 1
                        for h in h2:
                            if h in o1 and h in o2:
                                overlap_score[1] += 1
                        if overlap_score[0] == overlap_score[1]:
                            self.dev[d].add_feature(1)
                            self.dev[d + 1].add_feature(1)
                        elif overlap_score[0] > overlap_score[1]:
                            self.dev[d].add_feature(1)
                            self.dev[d + 1].add_feature(0)
                        else:
                            self.dev[d].add_feature(0)
                            self.dev[d + 1].add_feature(1)
                    else:
                        self.dev[d].add_feature(0)
                        self.dev[d + 1].add_feature(0)

                    len_diff_h1 = abs(len(o1) - len(h1)) + abs(len(o2) - len(h1))
                    len_diff_h2 = abs(len(o1) - len(h2)) + abs(len(o2) - len(h2))
                    if len_diff_h1 > len_diff_h2:
                        self.dev[d].add_feature(0)
                        self.dev[d + 1].add_feature(1)
                    elif len_diff_h1 < len_diff_h2:
                        self.dev[d].add_feature(1)
                        self.dev[d + 1].add_feature(0)
                    else:
                        self.dev[d].add_feature(0)
                        self.dev[d + 1].add_feature(0)

                    vocabulary.extend(o1)
                    vocabulary.extend(o2)
                    vocabulary.extend(h1)
                    vocabulary.extend(h2)
                    vocabulary = list(set(vocabulary))

                    v1 = [0] * len(vocabulary)
                    v2 = [0] * len(vocabulary)
                    vh1 = [0] * len(vocabulary)
                    vh2 = [0] * len(vocabulary)

                    for token in o1:
                        v1[vocabulary.index(token)] += 1
                    for token in o2:
                        v2[vocabulary.index(token)] += 1
                    for token in h1:
                        vh1[vocabulary.index(token)] += 1
                    for token in h2:
                        vh2[vocabulary.index(token)] += 1

                    v1_len = 0
                    v2_len = 0
                    vh1_len = 0
                    vh2_len = 0

                    for i, j, k, r in zip(v1, v2, vh1, vh2):
                        if i:
                            v1_len += i * i
                        if j:
                            v2_len += j * j
                        if k:
                            vh1_len += k * k
                        if r:
                            vh2_len += r * r
                    v1_len = math.sqrt(v1_len)
                    v2_len = math.sqrt(v2_len)
                    vh1_len = math.sqrt(vh1_len)
                    vh2_len = math.sqrt(vh2_len)

                    cos_o1_h1 = sum([i * j for (i, j) in zip(v1, vh1)]) / (v1_len * vh1_len)
                    cos_o1_h2 = sum([i * j for (i, j) in zip(v1, vh2)]) / (v1_len * vh2_len)
                    if cos_o1_h1 > cos_o1_h2:
                        self.dev[d].add_feature(1)
                        self.dev[d + 1].add_feature(0)
                    elif cos_o1_h1 == cos_o1_h2 and cos_o1_h1 != 0:
                        self.dev[d].add_feature(1)
                        self.dev[d + 1].add_feature(1)
                    elif cos_o1_h1 < cos_o1_h2:
                        self.dev[d].add_feature(0)
                        self.dev[d + 1].add_feature(1)
                    else:
                        self.dev[d].add_feature(0)
                        self.dev[d + 1].add_feature(0)

                    cos_o2_h1 = sum([i * j for (i, j) in zip(v2, vh1)]) / (v2_len * vh1_len)
                    cos_o2_h2 = sum([i * j for (i, j) in zip(v2, vh2)]) / (v2_len * vh2_len)

                    if cos_o2_h1 > cos_o2_h2:
                        self.dev[d].add_feature(1)
                        self.dev[d + 1].add_feature(0)
                    elif cos_o2_h1 == cos_o2_h2 and cos_o2_h1 != 0:
                        self.dev[d].add_feature(1)
                        self.dev[d + 1].add_feature(1)
                    elif cos_o2_h1 < cos_o2_h2:
                        self.dev[d].add_feature(0)
                        self.dev[d + 1].add_feature(1)
                    else:
                        self.dev[d].add_feature(0)
                        self.dev[d + 1].add_feature(0)

    def get_labels(self, data='train'):
        if data == 'train':
            return [i.label for i in self.train]
        else:
            return [i.label for i in self.dev]

    def get_features(self, data='train'):
        if data == 'train':
            return [i.F for i in self.train]
        else:
            return [i.F for i in self.dev]

    def naive_predictions(self, data='train'):
        dim = len(self.train) if data == 'train' else len(self.dev)
        pred = [0]*dim
        for i in range(dim):
            if i%2 == 0:
                pred[i] = 1
                pred[i+1] = 0
        return pred
    def random_predictions(self, data='train'):
        dim = len(self.train) if data == 'train' else len(self.dev)
        pred = [0] * dim
        for i in range(dim):
            if i%2 == 0:
                res = random.choice([1, 0])
                pred[i] = res
                pred[i + 1] = 1 - res
        return pred


def main():
    print('Start ingesting data')
    time_1 = time.time()

    c = Learning()
    c.ingest_data()

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print('Start extracting features..')
    for i in c.train:
        i.feature_extraction()
    for i in c.dev:
        i.feature_extraction()
    c.generate_cross_features(data='train')
    c.generate_cross_features(data='dev')

    train_features = c.get_features()
    train_labels = c.get_labels()

    dev_features = c.get_features(data='dev')
    dev_lables = c.get_labels(data='dev')

    print(train_features[:5])
    print(dev_features[:5])

    time_3 = time.time()
    print('feature extraction cost ', time_3 - time_2, ' second', '\n')


    print('Start training')
    p = Perceptron()
    p.train(train_features, train_labels)

    time_4 = time.time()
    print('training cost ', time_4 - time_3, ' second', '\n')

    print('Start predicting')
    p.predict(dev_features)
    print(p.prediction)
    time_5 = time.time()
    print('predicting cost ', time_5 - time_4, ' second', '\n')

    print('Score on our baseline model  on dev set::')
    e = Utils(p.prediction, dev_lables)
    score =e.evaluation()
    print("The accruacy socre is ", score)

    p.predict(train_features)
    print(p.prediction)
    print('Score on our baseline model on training set:')
    e = Utils(p.prediction, train_labels)
    score =e.evaluation()
    print("The accruacy socre is ", score)

    print('Score on the naive predictions compatibale with dev:')
    e = Utils(c.naive_predictions(data = 'dev'), dev_lables)
    score =e.evaluation()
    print("The accruacy socre is ", score)

    print('Score on the naive predictions compatibale with training:')
    e = Utils(c.naive_predictions(), train_labels)
    score =e.evaluation()
    print("The accruacy socre is ", score)

    print('Score on the random predicitons compatibale with dev:')
    e = Utils(c.random_predictions(), dev_lables)
    score =e.evaluation()
    print("The accruacy socre is ", score)

    print('Score on the random predicitons compatibale with training')
    e = Utils(c.random_predictions(data = 'dev'), train_labels)
    score =e.evaluation()
    print("The accruacy socre is ", score)




if __name__ == "__main__":
    # execute only if run as a script
    main()

