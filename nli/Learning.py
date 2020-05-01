from KB import KB
from Utils import  Utils
from Perceptron import  Perceptron
import jsonlines

class Learning:

    def __init__(self):
        self.train = []
        self.dev  = []
        self.train_data_path = 'jsonl/train.jsonl'
        self.dev_data_path = 'jsonl/dev.jsonl'
        self.train_label_path = 'jsonl/train-labels.lst'
        self.dev_label_path = 'jsonl/dev-labels.lst'

    def ingestData(self):
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

    def getLabels(self, data='train'):
        if data == 'train':
            return [i.label for i in self.train]
        else:
            return [i.label for i in self.dev]

    def mimicPredictions(self):
        pred = []
        for i in range(len(self.train)):
            if i%2 == 0:
                pred.append(1)
            else:
                pred.append(0)
        return pred

def main():
    Classifier = Learning()
    Classifier.ingestData()
    Eva = Utils(Classifier.mimicPredictions(),Classifier.getLabels(data = 'train'))
    Eva.Evaluation()

if __name__ == "__main__":
    # execute only if run as a script
    main()

