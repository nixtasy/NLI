import numpy as np
import jsonlines

class Evaluation:
    def Precision(self,TP, FP):
        if TP == 0:
            return 0
        else:
            return TP / (TP + FP)

    def Recall(self,TP, FN):
        if TP == 0:
            return 0
        else:
            return TP / (TP + FN)

    def F(self,P, R):
        if P == 0:
            return 0
        else:
            return 2 * P * R / (P + R)

    def Accuracy(self,TP, TN, FP, FN):
        return (TP + TN) / (TP + TN + FP + FN)

    def MimicPredictions(self):
        with jsonlines.open('jsonl/train.jsonl') as reader:
            train = [obj for obj in reader]
        pseudo_predictions = [1 for i in range(len(train))]
        pseudo_predictions[:10]
        return pseudo_predictions

    def Eva(self, Hn, predictions, labels):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for p, l in zip(predictions, labels):
            if p == Hn:
                if l == p:
                    TP += 1
                else:
                    FP += 1
            else:
                if l == p:
                    TN += 1
                else:
                    FN += 1
        P = self.Precision(TP, FP)
        R = self.Recall(TP, FN)
        F1 = self.F(P, R)
        Acc = self.Accuracy(TP, TN, FP, FN)
        print("H%d: Precison %f, Recall %f, Accuracy %f, F1 %f" % (Hn, P, R, Acc, F1))



class   BaselineClassifier(Evaluation):
    def __init__(self, fit_intercept=True):
        pass

    def __repr__(self):
        return "I am a Linear Regression model!"

    def ingest_data(self,X,y):
        """
       Ingests the given data

        Arguments:
        X: 1D or 2D numpy array
        y: 1D numpy array
        """
        with jsonlines.open('jsonl/train.jsonl') as reader:
            train = [obj for obj in reader]

        with jsonlines.open('jsonl/train-labels.lst') as reader:
            train_labels = [obj for obj in reader]

        with jsonlines.open('jsonl/dev.jsonl') as reader:
            dev = [obj for obj in reader]

        with jsonlines.open('jsonl/dev-labels.lst') as reader:
            dev_labels = [obj for obj in reader]

            # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # features and data
        self.features_ = X
        self.target_ = y
        return train_labels
# test_prediction = Evaluation
# test_train_labels =  BaselineClassifier
# testresult = Evaluation.Eva(1, test_prediction.MimicPredictions(), test_train_labels.ingest_data())

testEva = Evaluation()
Baseline = BaselineClassifier()
prediction = Baseline.MimicPredictions()
testEva.Eva(1, prediction, Baseline.ingest_data())