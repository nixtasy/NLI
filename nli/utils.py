def Precision(TP, FP):
    if TP == 0:
        return 0
    else:
        return TP / (TP + FP)


def Recall(TP, FN):
    if TP == 0:
        return 0
    else:
        return TP / (TP + FN)


def F(P, R):
    if P == 0:
        return 0
    else:
        return 2 * P * R / (P + R)


def Accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)


def Eva(Hn, predictions, labels):
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
    P = Precision(TP, FP)
    R = Recall(TP, FN)
    F1 = F(P, R)
    Acc = Accuracy(TP, TN, FP, FN)
    print("H%d: Precison %f, Recall %f, Accuracy %f, F1 %f" % (Hn, P, R, Acc, F1))