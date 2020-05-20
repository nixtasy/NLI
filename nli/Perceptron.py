import random
class Perceptron:
    def __init__(self):
        self.weight = None
        self.prediction = None
        self.learning_rate = 0.6
        self.overflow_batch_size = 4096
        self.max_epoch = 20
        self.weight_init_seed = 1
        self.decay_rate = 0.9

    def feedforward(self, x):
        return sum([i * j for (i, j) in zip(x, self.weight)])

    def predict(self, features):
        res = []
        for feature in features:
            x = [1]
            x.extend(feature)
            res.append(1 if self.feedforward(x) > 0 else 0)
        self.prediction = res

    def train(self, features, labels):
        self.weight = [self.weight_init_seed] * (len(features[0]) + 1) # add bias in the beginning
        for epo in range(self.max_epoch):
            # start_state = random.getstate()
            # random.shuffle(features)
            # random.setstate(start_state)
            # random.shuffle(labels)
            """
                Standard Gradient Descent
            """
            # for index, feature in enumerate(features):
            #     x = [1] # add bias
            #     x.extend(feature)
            #     y = labels[index]
            #     y_hat = self.feedforward(x)
            #     if y == 1:
            #         if y_hat < 0:
            #             for i in range(len(self.weight)):
            #                 self.weight[i] += self.learning_rate * x[i]
            #     else:
            #         if y_hat > 0:
            #             for i in range(len(self.weight)):
            #                 self.weight[i] -= self.learning_rate * x[i]

            """
            
               Adapted Mini-batch Stochastic Gradient Descent with Learning Rate Decay
            
            """
            correct_counts = 0
            while correct_counts < self.overflow_batch_size:
                self.learning_rate *= self.decay_rate
                # index = np.random.randint(0, len(labels) - 1)
                index = random.randint(0,len(labels)-1)
                x = [1]  # add bias
                x.extend(features[index])
                y = labels[index]
                y_hat = self.feedforward(x)
                if y == 1:
                    if y_hat < 0:
                        for i in range(len(self.weight)):
                            self.weight[i] += self.learning_rate * x[i]
                    else:
                        correct_counts += 1
                else:
                    if y_hat > 0:
                        for i in range(len(self.weight)):
                            self.weight[i] -= self.learning_rate * x[i]
                    else:
                        correct_counts += 1

            print("finish epoch ", epo , "..")








