import sys
import os
import time
import numpy

slot_split = "\t"
term_split = ":"

def sign(a):
    if a >= 0:
        return 1
    else:
        return -1

def dot_product(a, b):
    return numpy.dot(a, b) # TODO: add try catch, add branch: not using numpy

def score(weight, feature):
    return dot_product(weight, feature)

def predict(model_weight, dim):
    right = 0
    wrong = 0
    sum = 0

    for line in sys.stdin:
        sum += 1
        feature, label = readdata(line, dim)
        if None == feature:
            continue

        modelscore = score(model_weight, feature)
        modelpredict = sign(modelscore)
        print >> sys.stderr, time.ctime(), "DEBUG, modelscore:", modelscore, "modelpredict:", modelpredict, "label:", label, 
        print str(modelscore) + "\t" + str(modelpredict) + "\t" + str(label)
        if modelpredict == label:
            print >> sys.stderr, "right"
            right += 1
        else:
            print >> sys.stderr, "wrong"
            wrong += 1

    failed = sum - right - wrong
    if 0 != failed:
        print >> sys.stderr, time.ctime(), "INFO, passed_num:", sum - right - wrong
    print >> sys.stderr, time.ctime(), "INFO, right:", right, "wrong:", wrong, "total:", right + wrong, "accuracy:", right * 1.0 / (right + wrong)

def loadmdoel(model_file):
    f = open(model_file)

    weight = []
    count = 0
    for line in f:
        line = line.strip()
        term = line.split()
        if not len(term) == 1:
            print >> sys.stderr, time.ctime(), "ERROR, not len(term) == 1, len(term):", len(term) 
        term = term[0]

        if 0 == count:
            dim = int(term)
        else:
            weight.append(term)
        count += 1

    weight = map(float, weight)

    if dim == len(weight):
        print >> sys.stderr, time.ctime(), "INFO, read done. dim:", dim
    else:
        print >> sys.stderr, time.ctime(), "ERROR, dim != len_of_weight in loadmdoel.", dim, "!=", len(weight)

    f.close()

    return weight

def readdata(line, expect_dim):
    line = line.strip()
    slots = line.split(slot_split)
    label = int(slots[0])
    length = len(slots[1:])

    if length != expect_dim:
        print >> sys.stderr, time.ctime(), "WARNING, length != expect_dim.", length, "!=", dim, " will continue"
        return None, None
    
    features = []
    for term in slots[1:]:
        id, feature = term.split(term_split)
        features.append(feature)
    features = map(float, features)

    return features, label

def main(argv):
    if len(argv) < 2:
        print >> sys.stderr, time.ctime(), "Usage:", argv[0], "model_file"
        return
        
    model_file = argv[1]

    weight = loadmdoel(model_file)

    predict(weight, len(weight))

if "__main__" == __name__:
    main(sys.argv)
