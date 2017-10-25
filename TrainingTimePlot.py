import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import svm
import DataProcessor
import SVMTraining

train_file, dev_file = "income-data/income.train.txt.5k", "income-data/income.dev.txt"

feature2index = DataProcessor.create_feature_map(train_file)
train_data, train_target = DataProcessor.map_data(train_file, feature2index)

trainingExamples = [5, 50, 500, 5000]
trainingTime = []

for i in range(len(trainingExamples)):
    _, trainTime = SVMTraining.SVM_fit(train_data[:trainingExamples[i]], train_target[:trainingExamples[i]])
    trainingTime.append(trainTime)
    
    print("End of run with examples = " + str(trainingExamples[i]))

plt.plot(trainingExamples, trainingTime)
plt.axis([0, 5000, 0, 4])
plt.xlabel('Number of Training Examples')
plt.ylabel('Training Time')
plt.show()
