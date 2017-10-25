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

z = np.polyfit(trainingExamples, trainingTime, 2)
f = np.poly1d(z)

x_new = np.linspace(trainingExamples[0], trainingExamples[-1], 50)
y_new = f(x_new)

plt.plot(trainingExamples, trainingTime, 'rx', label='data')
plt.plot(x_new, y_new, 'b-', label='fit')
plt.axis([0, 5000, 0, 4])
plt.xlabel('Number of Training Examples')
plt.ylabel('Training Time')
plt.show()
