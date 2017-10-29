import numpy as np
import time
from Predictor import Predictor
import matplotlib.pyplot as plt
#from sklearn import svm
import DataProcessor

## Utilize sklearn's SVM program to classify individuals earning less
## or more than 50K/year.

def DevError(weightVector, bias, data, target):
 
        i = 0
        positiveResults = 0
        errors = 0

        for row in data:
            #row = row
            if target[i] * (np.dot(row, weightVector) + bias) <= 0:
                # <= 50
               errors += 1
            
            i += 1

        error_rate = errors / len(data)        
        return error_rate

def Pegasos(data, target, devData, devTarget):

    nCount = len(data)
    weightVector = np.zeros(len(train_data[0]))
    bias = 0
    cParam = 1
    lambdaVar = 2 / (nCount * cParam)
    currentTrainingCount = 1
    epochCount = 1
    totalEpoch = 250
##    supportVectors = 0
    startTime = time.time()
    objective = []
    train_Error = []
    dev_Error = []
    epoch = []

    while epochCount <= totalEpoch:

          errors = 0
          for i in range(0, nCount):
       
              learningRate = 1 / (lambdaVar * currentTrainingCount)

              if target[i] * (np.dot(data[i], weightVector) + bias) < 1:

                  weightVector = weightVector - learningRate * (lambdaVar * weightVector - target[i] * data[i])
                  bias = bias - learningRate * (lambdaVar * bias - target[i])
##                  supportVectors += 1

              else:
                  weightVector = weightVector - learningRate * lambdaVar * weightVector
                  bias = bias - learningRate * lambdaVar * bias
              
              currentTrainingCount += 1

          endTime = time.time()       
          epochCount += 1

          trainError = DevError(weightVector, bias, data, target)
          devError = DevError(weightVector, bias, devData, devTarget)
          
          print("The dev error rate is {:.2%} for epoch = {:1d}".format(devError, epochCount))

          print("The Training error rate is {:.2%} for epoch = {:1d} ".format(trainError, epochCount))
      
          print("The Pegasos ran for {:.2f} seconds".format((endTime - startTime)))

##          print(str(supportVectors))

          objective.append(objective_function(weightVector, bias, data, target, lambdaVar, cParam))
          train_Error.append(100 * trainError)
          dev_Error.append(100 * devError)
          epoch.append(epochCount - 1)
    
    return weightVector, objective, train_Error, dev_Error, epoch

def objective_function(model, bias, train_data, target, lambdaVar, cParam = 1):
    totalSlack = 0
    N = len(train_data)
    for j in range(len(train_data)):
        slack = 1 - target[j] * (model.dot(train_data[j]) + bias)
        if slack > 0:
            totalSlack += slack
        
    objective = 0.5 * lambdaVar * (model.dot(model.T) + bias * bias) + (1 / N) * totalSlack

    print("The minimized objective function value is {:.5f}".format(objective))
    return objective

def test(dataSet, data, model, bias, cParam = 1):
    errors = sum(y * (model.dot(vecx) + bias) <= 0 for vecx, y in data)
    error_rate = errors / len(data)

    print("The", dataSet, "error rate is {:.2%} for C = {:.2f%}".format(error_rate[0], cParam))


if __name__ == "__main__":
    train_file, dev_file = "income-data/income.train.txt.5k", "income-data/income.dev.txt"

    feature2index = DataProcessor.create_feature_map(train_file)
    train_data, train_target = DataProcessor.map_data(train_file, feature2index)
    dev_data, dev_target = DataProcessor.map_data(dev_file, feature2index)

    weightVector, objective, trainError, devError, epoch = Pegasos(train_data, train_target, dev_data, dev_target)

    plt.plot(epoch, trainError, 'r-', epoch, devError, 'k-')
    plt.legend(('Training Set', 'Dev Set'))
    plt.axis([0, 250, 15, 25])
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate, %')
    plt.show()

    plt.plot(epoch, objective, 'k-')
    plt.axis([0, 250, 0.3, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Objective Function')
    plt.show()
