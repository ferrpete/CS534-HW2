import numpy as np
import time
from sklearn import svm
import DataProcessor

## Utilize sklearn's SVM program to classify individuals earning less
## or more than 50K/year.

def SVM_fit(data, target, cParam = 1, _kernel='linear', _degree=1, _coef0=0):

    clf = svm.SVC(kernel = _kernel, degree = _degree, C = cParam, coef0 = _coef0)

    startTime = time.time()
    clf.fit(data, target)
    endTime = time.time()
    print("The SVM ran for %s seconds" % (endTime - startTime))
    print("The number of support vectors are: ", str(len(clf.support_vectors_)))

    return clf, (endTime - startTime)

def test(dataSet, data, model, bias, cParam = 1):
    errors = sum(y * (model.dot(vecx) + bias) <= 0 for vecx, y in data)
    error_rate = errors / len(data)

    print("The", dataSet, "error rate is {:.2%} for C = {:.2f}".format(error_rate[0], cParam))

def margin_violation(model, bias, support_vectors, target, cParam = 1):

    marginViolation = 0

    for i, vecx in enumerate(support_vectors):
        if 1 - target[i] * (model.dot(vecx) + bias) > 0:
            marginViolation += 1

    print("The number of support vectors with margin violations is: {:.0f}".format(marginViolation))

def objective_function(model, bias, train_data, target, cParam = 1):
    totalSlack = 0
    for j in range(len(train_data)):
        slack = 1 - target[j] * (model.dot(train_data[j]) + bias)
        if slack > 0:
            totalSlack += slack
        
    objective = 0.5 * (model.dot(model.T) + bias * bias) + cParam * totalSlack

    print("The total slack is: {:.2f}".format(totalSlack[0]))
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

    _kernel = int(input("Kernel [1: linear | 2: quadratic] > "))
    kernel = 'linear'
    degree = 1
    coef0 = 0
    if _kernel == 2:
        kernel = 'poly'
        degree = 2
        coef0 = 1

    while True:

        cParam = float(input("c Parameter > "))

        clf, _ = SVM_fit(train_data, train_target, cParam, kernel, degree, coef0)
        train_predict = clf.score(train_data, train_target)
        dev_predict = clf.score(dev_data, dev_target)
        train_error = 1 - train_predict
        print("The training error rate is {:.2%} for C = {:.2f}".format(train_error, cParam))
        
        dev_error = 1 - dev_predict
        print("The dev error rate is {:.2%} for C = {:.2f}".format(dev_error, cParam))

        if _kernel == 1:
            margin_violation(clf.coef_, clf.intercept_, clf.support_vectors_, train_target, cParam)
            objective_function(clf.coef_, clf.intercept_, train_data, train_target, cParam)
            most_violated(clf.coef_, clf.intercept_, train_data, train_target, feature2index)
