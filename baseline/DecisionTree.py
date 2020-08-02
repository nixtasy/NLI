from sklearn import tree

class DecisionTree:
    def __init__(self):
        self.prediction = None
        self.clf = tree.DecisionTreeClassifier()

    def predict(self, features):
        self.prediction = self.clf.predict(features)

    def train(self, features, labels):
        self.clf.fit(features, labels)








