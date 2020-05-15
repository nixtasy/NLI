from KB import KB
from Utils import  Utils
from Perceptron import  Perceptron
import jsonlines
import random
import math
import numpy as np

class Learning:

    def __init__(self):
        self.train = []
        self.dev  = []
        self.train_data_path = 'jsonl/train.jsonl'
        self.dev_data_path = 'jsonl/dev.jsonl'
        self.train_label_path = 'jsonl/train-labels.lst'
        self.dev_label_path = 'jsonl/dev-labels.lst'
        self.tfmatrix = []

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

    def generate_cross_features(self,data='train'):
        if data == 'train':
            for t in range(len(self.train)):
                vocabulary = []
                if t % 2 == 0:
                    o1 = [token.lower() for token in self.train[t].o1.split()]
                    o2 = [token.lower() for token in self.train[t].o2.split()]
                    h1 = [token.lower() for token in self.train[t].h.split()]
                    h2 = [token.lower() for token in self.train[t+1].h.split()]

                    o1[-1] = o1[-1][:-1]
                    o2[-1] = o2[-1][:-1]
                    h1[-1] = h1[-1][:-1]
                    h2[-1] = h2[-1][:-1]

                    vocabulary.extend(o1)
                    vocabulary.extend(o2)
                    vocabulary.extend(h1)
                    vocabulary.extend(h2)
                    vocabulary = list(set(vocabulary))

                    v1 = [0 for i in range(len(vocabulary))]
                    v2 = [0 for i in range(len(vocabulary))]
                    vh1 = [0 for i in range(len(vocabulary))]
                    vh2 = [0 for i in range(len(vocabulary))]

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

                    for i,j,k,r in zip(v1,v2,vh1,vh2):
                        if i:
                            v1_len += i*i
                        if j:
                            v2_len += j*j
                        if k:
                            vh1_len += k*k
                        if r:
                            vh2_len += r*r
                    v1_len = math.sqrt(v1_len)
                    v2_len = math.sqrt(v2_len)
                    vh1_len = math.sqrt(vh1_len)
                    vh2_len = math.sqrt(vh2_len)

                    cos_o1_h1 = sum([i * j for (i, j) in zip(v1, vh1)]) / (v1_len * vh1_len)
                    cos_o1_h2 = sum([i * j for (i, j) in zip(v1, vh2)]) / (v1_len * vh2_len)

                    if cos_o1_h1 > cos_o1_h2:
                        self.train[t].add_feature(1)
                        self.train[t+1].add_feature(0)
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
                        self.train[t+1].add_feature(0)
                    elif cos_o2_h1 == cos_o2_h2 and cos_o2_h1 != 0:
                        self.train[t].add_feature(1)
                        self.train[t + 1].add_feature(1)
                    elif cos_o2_h1 < cos_o2_h2:
                        self.train[t].add_feature(0)
                        self.train[t + 1].add_feature(1)
                    else:
                        self.train[t].add_feature(0)
                        self.train[t + 1].add_feature(0)
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

                    vocabulary.extend(o1)
                    vocabulary.extend(o2)
                    vocabulary.extend(h1)
                    vocabulary.extend(h2)
                    vocabulary = list(set(vocabulary))

                    v1 = [0 for i in range(len(vocabulary))]
                    v2 = [0 for i in range(len(vocabulary))]
                    vh1 = [0 for i in range(len(vocabulary))]
                    vh2 = [0 for i in range(len(vocabulary))]

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
                        self.dev[d+1].add_feature(0)
                    elif cos_o2_h1 == cos_o2_h2 and cos_o2_h1 != 0:
                        self.dev[d].add_feature(1)
                        self.dev[d+1].add_feature(1)
                    elif cos_o2_h1 < cos_o2_h2:
                        self.dev[d].add_feature(0)
                        self.dev[d+1].add_feature(1)
                    else:
                        self.dev[d].add_feature(0)
                        self.dev[d+1].add_feature(0)
        # print('Cross features successfully generated ..')


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

    def mimic_predictions(self):
        pred = []
        for i in range(len(self.train)):
            if i%2 == 0:
                pred.append(1)
            else:
                pred.append(0)
        return pred

    def tf_matrix(self, data='train'):
        if data == 'train':
            vocab = self.get_vocabulary(data)
            for i in self.train:
                F = [0 for i in range(len(vocab))]
                tokens_o1 = [token.lower() for token in i.o1.split()]
                tokens_o2 = [token.lower() for token in i.o2.split()]
                tokens_h = [token.lower() for token in i.h.split()]
                tokens_o1[-1] = tokens_o1[-1][:-1]
                tokens_o2[-1] = tokens_o2[-1][:-1]
                tokens_h[-1] = tokens_h[-1][:-1]
                for token in tokens_o1:
                    F[vocab.index(token)] += 1
                for token in tokens_o2:
                    F[vocab.index(token)] += 1
                for token in tokens_h:
                    F[vocab.index(token)] -= 2
                self.tfmatrix.append(F)
        else:
            vocab = self.get_vocabulary(data)
            for i in self.dev:
                F = [0 for i in range(len(vocab))]
                tokens_o1 = [token.lower() for token in i.o1.split()]
                tokens_o2 = [token.lower() for token in i.o2.split()]
                tokens_h = [token.lower() for token in i.h.split()]
                tokens_o1[-1] = tokens_o1[-1][:-1]
                tokens_o2[-1] = tokens_o2[-1][:-1]
                tokens_h[-1] = tokens_h[-1][:-1]
                for token in tokens_o1:
                    F[vocab.index(token)] += 1
                for token in tokens_o2:
                    F[vocab.index(token)] += 1
                for token in tokens_h:
                    F[vocab.index(token)] -= 2
                self.tfmatrix.append(F)

    def get_vocabulary(self):
        tokens = []
        for i in self.train:
            tokens.extend([token.lower() for token in i.o1.split()])
            tokens[-1] = tokens[-1][:-1]
            tokens.extend([token.lower() for token in i.o2.split()])
            tokens[-1] = tokens[-1][:-1]
            tokens.extend([token.lower() for token in i.h.split()])
            tokens[-1] = tokens[-1][:-1]

        for i in self.dev:
            tokens.extend([token.lower() for token in i.o1.split()])
            tokens[-1] = tokens[-1][:-1]
            tokens.extend([token.lower() for token in i.o2.split()])
            tokens[-1] = tokens[-1][:-1]
            tokens.extend([token.lower() for token in i.h.split()])
            tokens[-1] = tokens[-1][:-1]
        return list(set(tokens))


def main():
    Classifier = Learning()
    Classifier.ingest_data()

    for i in Classifier.train:
        i.feature_extraction()
    for i in Classifier.dev:
        i.feature_extraction()

    Classifier.generate_cross_features(data='train')
    Classifier.generate_cross_features(data='dev')

    features = Classifier.get_features()
    lables = Classifier.get_labels()
    lr  = [0.003,0.006,0.009,0.3,0.6,0.9]
    for llrr in lr:
        print("==============================", llrr, "==========================================")


        # start_state = random.getstate()
        # random.shuffle(features)
        # random.setstate(start_state)
        # random.shuffle(lables)

        session = Perceptron(features[0], lables[0], learning_rate = llrr)
        for i in range(len(features)-1):
            session.train()
            session.feed(features[i+1],lables[i+1])
        W = session.weights

        dev_features = Classifier.get_features(data='dev')
        dev_lables = Classifier.get_labels(data='dev')
        session = Perceptron(dev_features[0], dev_lables[0], W)
        # session = Perceptron(features[0], lables[0], W)
        print(dev_features[:100])
        predictions = []
        for i in range(len(dev_features) - 1):
            predictions.append(session.predict(result=True))
            session.feed(dev_features[i + 1], dev_lables[i + 1])
        print(predictions)
        Eva = Utils(predictions, dev_lables)
        # print(len(dev_features[0]))

        Eva.Evaluation()


if __name__ == "__main__":
    # execute only if run as a script
    main()

