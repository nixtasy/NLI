class Perceptron:
    def __init__(self, learning_rate = 0.0006, max_iteration = 10):
        self.weight = None
        self.prediction = []
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration

    def feedforward(self, x):
        return sum([i * j for (i, j) in zip(x, self.weight)])

    def predict(self, features):
        for feature in features:
            x = [1]
            x.extend(feature)
            self.prediction.append(1 if self.feedforward(x) > 0 else 0)

    def train(self, features, labels):
        self.weight = [1 - 0] * (len(features[0]) + 1) # add bias in the beginning
        for iter in range(self.max_iteration):
            for index, feature in enumerate(features):
                x = [1] # add bias
                x.extend(feature)
                y = labels[index]
                y_hat = self.feedforward(x)
                if y == 1:
                    if y_hat < 0:
                        for i in range(len(self.weight)):
                            self.weight[i] += self.learning_rate * x[i]
                else:
                    if y_hat > 0:
                        for i in range(len(self.weight)):
                            self.weight[i] -= self.learning_rate * x[i]
            print("finish epoch ",iter,"..")


