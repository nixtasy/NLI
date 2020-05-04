from KB import KB
from Utils import  Utils
from Perceptron import  Perceptron
import jsonlines
import random

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

        with jsonlines.open(self.dev_data_path) as reader:
            dev_data = [obj for obj in reader]

        with jsonlines.open(self.dev_label_path) as reader:
            dev_label = [obj for obj in reader]

        for X, Y in zip(dev_data, dev_label):
            self.dev.append(KB(X['obs1'],X['obs2'],X['hyp1'],1 if Y == 1 else 0))
            self.dev.append(KB(X['obs1'], X['obs2'], X['hyp2'], 1 if Y == 2 else 0))

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
    # vocab = Classifier.get_vocabulary()
    # print(len(vocab))
    for i in Classifier.train:
        i.feature_extraction()
    for i in Classifier.dev:
        i.feature_extraction()
    # print(Classifier.get_features())
    # with open('train_F.txt', 'w') as file:
    #     file.writelines((str(i[0])+'\t'+str(i[1])+'\n' for i in Classifier.get_features()))
    features = Classifier.get_features()
    lables = Classifier.get_labels()


    start_state = random.getstate()
    random.shuffle(features)
    random.setstate(start_state)
    random.shuffle(lables)

    session = Perceptron(features[0], lables[0])
    for i in range(len(features)-1):
        session.train()
        session.feed(features[i+1],lables[i+1])
    W = session.weights

    dev_features = Classifier.get_features(data='dev')
    dev_lables = Classifier.get_labels(data='dev')
    session = Perceptron(dev_features[0], dev_lables[0], W)
    predictions = []
    for i in range(len(dev_features) - 1):
        predictions.append(session.predict(result=True))
        session.feed(dev_features[i + 1], dev_lables[i + 1])

    Eva = Utils(predictions, dev_lables)
    Eva.Evaluation()


    # Classifier.tf_matrix('train')
    # with open('train_F.txt', 'w') as file:
    #     file.writelines((str(i.F)+'\n' for i in Classifier.train))
    # Eva = Utils(Classifier.mimic_predictions(),Classifier.get_labels(data = 'train'))
    # Eva.Evaluation()

if __name__ == "__main__":
    # execute only if run as a script
    main()

