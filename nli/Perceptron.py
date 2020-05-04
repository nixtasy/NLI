class Perceptron:
    def __init__(self, input, label, weights = [1, 0.1, 0.1]):
        self.label = label
        self.input = [1]
        self.input.extend(input)
        self.weights = weights
        self.prediction = []

    def predict(self, result = False):
        self.prediction = sum([i * j for (i, j) in zip(self.input, self.weights)])
        if result:
            return 1 if self.prediction>=0 else 0

    def feed(self, input, label):
        self.label = label
        self.input = [1]
        self.input.extend(input)

    def train(self):
        self.predict()
        if self.label == 1:
            if self.prediction < 0:
                for i in range(len(self.weights)):
                    self.weights[i] += self.input[i]
        else:
            if self.prediction > 0:
                for i in range(len(self.weights)):
                    self.weights[i] -= self.input[i]

