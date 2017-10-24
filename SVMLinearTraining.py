import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import svm
import HW1Reference

## Utilize sklearn's SVM program to classify individuals earning less
## or more than 50K/year.

def SVM_fit(train_data, dev_data, cParam = 1):

    featureData = []
    target = []

    for j, (vecx, y) in enumerate(train_data, 1):
        featureData.append(vecx)
        target.append(y)

    clf = svm.SVC(kernel = 'linear', C = cParam)

    startTime = time.time()
    clf.fit(featureData, target)
    print("The SVM ran for %s seconds" % (time.time() - startTime))
    print("The number of support vectors are: ", str(len(clf.support_vectors_)))

    return clf

def margin_violation(clf, cParam = 1):

    marginViolation = 0

    for i, dual in enumerate(clf.dual_coef_[0]):
        if abs(dual) == cParam:
            marginViolation += 1

    print("The number of margin violations is: ", str(marginViolation))

if __name__ == "__main__":
    train_file, dev_file = "income-data/income.train.txt.5k", "income-data/income.dev.txt"

    feature2index = HW1Reference.create_feature_map(train_file)
    train_data = HW1Reference.map_data(train_file, feature2index)
    dev_data = HW1Reference.map_data(dev_file, feature2index)

    clf = SVM_fit(train_data, dev_data)
    margin_violation(clf)
