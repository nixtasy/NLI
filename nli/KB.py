import re
import math
class KB:

    def __init__(self,o1,o2,h,label):
        self.o1 = o1
        self.o2 = o2
        self.h = h
        self.label = label
        self.F = []  #feature list

    def feature_extraction(self):
        vocabulary = []
        tokens_o1 = [token.lower() for token in self.o1.split()]
        tokens_o2 = [token.lower() for token in self.o2.split()]
        tokens_h = [token.lower() for token in self.h.split()]

        tokens_o1[-1] = tokens_o1[-1][:-1]
        tokens_o2[-1] = tokens_o2[-1][:-1]
        tokens_h[-1] = tokens_h[-1][:-1]

        vocabulary.extend(tokens_o1)
        vocabulary.extend(tokens_o2)
        vocabulary.extend(tokens_h)

        v1 = [0 for i in range(len(vocabulary))]
        v2 = [0 for i in range(len(vocabulary))]
        vh = [0 for i in range(len(vocabulary))]

        for token in tokens_o1:
            v1[vocabulary.index(token)] += 1
        for token in tokens_o2:
            v2[vocabulary.index(token)] += 1
        for token in tokens_h:
            vh[vocabulary.index(token)] += 1

        v1_len = 0
        v2_len = 0
        vh_len = 0
        for i,j,k in zip(v1,v2,vh):
            if i:
                v1_len += i*i
            if j:
                v2_len += j*j
            if k:
                vh_len += k*k
        v1_len = math.sqrt(v1_len)
        v2_len = math.sqrt(v2_len)
        vh_len = math.sqrt(vh_len)

        self.F.append(sum([i*j for (i, j) in zip(v1, vh)]) / (v1_len * vh_len))
        self.F.append(sum([i*j for (i, j) in zip(v2, vh)]) / (v2_len * vh_len))
        # self.F = [0,0]
        # for i in range(len(v1)):
        #     self.F[0] += v1[i] * vh[i]
        #     self.F[1] += v2[i] * vh[i]


        # print(v1)
        # print(v2)
        # print(vh)
        # print(self.F)