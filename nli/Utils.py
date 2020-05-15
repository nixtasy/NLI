class Utils:

    def __init__(self, predictions, labels):
        self.predictions = predictions # Predictions, either from model output or mimic data
        self.labels = labels # Ground Truths
        self.tp = 0 # True Positives
        self.tn = 0 # True Negatives
        self.fp = 0 # False Positives
        self.fn = 0 # False Negatives
        self.P = 0 # Precision
        self.R = 0 # Recall
        self.A = 0 # Accuracy
        self.F = 0 # F-1 Score

    def precision(self):
        if self.tp == 0:
            self.P = 0
        else:
            self.P = self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp == 0:
            self.R = 0
        else:
            self.R = self.tp / (self.tp + self.fn)

    def f1(self):
        if self.P == 0:
            self.F = 0
        else:
            self.F =  2 * self.P * self.R / (self.P + self.R)

    def accuracy(self):
        self.A = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def evaluation(self):
        for p, l in zip(self.predictions, self.labels):
            if p == 1:
                if l == 1:
                    self.tp += 1
                else:
                    self.fp += 1
            else:
                if l == 0:
                    self.tn += 1
                else:
                    self.fn += 1
        self.precision()
        self.recall()
        self.f1()
        self.accuracy()
        print("Precison %f, Recall %f, Accuracy %f, F1 %f" % (self.P, self.R, self.A, self.F))

