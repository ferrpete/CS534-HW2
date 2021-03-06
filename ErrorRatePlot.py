import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import DataProcessor
import SVMTraining

train_file, dev_file = "income-data/income.train.txt.5k", "income-data/income.dev.txt"

feature2index = DataProcessor.create_feature_map(train_file)
train_data, train_target = DataProcessor.map_data(train_file, feature2index)
dev_data, dev_target = DataProcessor.map_data(dev_file, feature2index)

cParam = [0.01, 1, 10, 20, 40, 60, 80, 100, 125, 150, 200, 250]
train_error = []
dev_error = []
_kernel = 'poly'
_degree = 2
_coef0 = 1

for i in range(len(cParam)):
    clf, _ = SVMTraining.SVM_fit(train_data, train_target, cParam[i], _kernel, _degree, _coef0)
    train_predict = clf.score(train_data, train_target)
    dev_predict = clf.score(dev_data, dev_target)
    trainError = 100 * (1 - train_predict)
    devError = 100 * (1 - dev_predict)

    train_error.append(trainError)
    dev_error.append(devError)
    print("End of run with C = {:.2f}".format(cParam[i]))

plt.plot(cParam, train_error, 'r-', cParam, dev_error, 'k--')
plt.legend(('Training Set', 'Dev Set'))
plt.axis([0, 250, 0, 30])
plt.xlabel('Penalty Parameter, C')
plt.ylabel('Error Rate, %')
plt.show()
