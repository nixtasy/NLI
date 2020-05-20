class KB:

    def __init__(self,o1,o2,h,label):
        self.o1 = o1
        self.o2 = o2
        self.h = h
        self.label = label
        self.F = []  #feature list


    def add_feature(self, feature):
        self.F.append(feature)

    def feature_extraction(self):

        tokens_o1 = [token.lower() for token in self.o1.split()]
        tokens_o2 = [token.lower() for token in self.o2.split()]
        tokens_h = [token.lower() for token in self.h.split()]

        tokens_o1[-1] = tokens_o1[-1][:-1]
        tokens_o2[-1] = tokens_o2[-1][:-1]
        tokens_h[-1] = tokens_h[-1][:-1]

        res = 0
        for h in tokens_h:
            if h in tokens_o1 and h in tokens_o2:
                res = 1
        self.F.append(res)


