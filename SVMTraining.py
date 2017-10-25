import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import svm
import DataProcessor

## Utilize sklearn's SVM program to classify individuals earning less
## or more than 50K/year.

def SVM_fit(train_data, train_target, cParam = 1, _kernel='linear', _degree=1):

    clf = svm.SVC(kernel = _kernel, degree = _degree, C = cParam)

    startTime = time.time()
    clf.fit(train_data, train_target)
    print("The SVM ran for %s seconds" % (time.time() - startTime))
    print("The number of support vectors are: ", str(len(clf.support_vectors_)))

    return clf

def test(dataSet, data, model, bias, cParam = 1):
    errors = sum(y * (model.dot(vecx) + bias) <= 0 for vecx, y in data)
    error_rate = errors / len(data)

    print("The", dataSet, "error rate is {:.2%} for C = {:.2f}".format(error_rate[0], cParam))

def margin_violation(clf, cParam = 1):

    marginViolation = 0

    for i, dual in enumerate(clf.dual_coef_[0]):
        if abs(dual) == cParam:
            marginViolation += 1

    print("The number of support vectors with margin violations is: {:.2f}".format(marginViolation))

def objective_function(model, bias, train_data, target, cParam = 1):
    totalSlack = 0
    for j in range(len(train_data)):
        slack = 1 - target[j] * (model.dot(train_data[j]) + bias)
        if slack > 0:
            totalSlack += slack
        
    objective = 0.5 * (model.dot(model.T) + bias * bias) + cParam * totalSlack

    print("The total amount of margin violation is: {:.2f}".format(totalSlack[0]))
    print("The minimized objective function value is {:.2f}".format(objective[0][0]))

def most_violated(model, bias, train_data, target, feature2index):
    slackArray = []

    for j in range(len(train_data)):
        slack = 1 - target[j] * (model.dot(train_data[j]) + bias)
        if slack > 0:
            slackArray.append(slack[0])

    slackArray = np.array(slackArray)
    violations = slackArray.argsort()[::-1]

    positiveViolation = []
    i = 0
    j = 0
    while i < 6:
        train_data[violations[j]][1]
        if target[j] == 1:
            positiveViolation.append(violations[j])
            i += 1
        j += 1

    negativeViolation = []
    i = 0
    j = 0
    while i < 6:
        if target[j] == -1:
            negativeViolation.append(violations[j])
            i += 1
        j += 1
            

    print("The most violated positive examples are :", str(positiveViolation),
          " with slacks of ", str(slackArray[positiveViolation]))
    print("The most violated negative examples are :", str(negativeViolation),
          " with slacks of ", str(slackArray[negativeViolation]))
    
if __name__ == "__main__":
    train_file, dev_file = "income-data/income.train.txt.5k", "income-data/income.dev.txt"

    feature2index = DataProcessor.create_feature_map(train_file)
    train_data, train_target = DataProcessor.map_data(train_file, feature2index)
    dev_data, dev_target = DataProcessor.map_data(dev_file, feature2index)

    _kernel = int(input("Kernel [1: linear | 2:poly] > "))
    kernel = 'linear'
    degree = 1
    if _kernel == 2:
        kernel = 'poly'
        degree = 2

    while True:

        cParam = float(input("c Parameter > "))

        clf = SVM_fit(train_data, train_target, cParam, kernel, degree)
        test = clf.predict(train_data)
        dev = clf.predict(dev_data)
        test_result = np.equal(test, train_target)
        dev_result = np.equal(dev, dev_target)
        print(str(dev_result))
 #       test('training', train_data, clf.coef_, clf.intercept_, cParam)
 #       test('dev', dev_data, clf.coef_, clf.intercept_, cParam)
        
        margin_violation(clf, cParam)
        objective_function(clf.coef_, clf.intercept_, train_data, train_target, cParam)
        most_violated(clf.coef_, clf.intercept_, train_data, train_target, feature2index)
