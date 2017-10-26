import numpy as np
import time
from Predictor import Predictor
#from sklearn import svm
import DataProcessor

## Utilize sklearn's SVM program to classify individuals earning less
## or more than 50K/year.

def DevError(weightVector, data, target):
 
        i = 0
        positiveResults = 0
        errors = 0

        for row in data:
            #row = row
            
            if target[i] * np.dot(row, weightVector) <= 0:
                # <= 50
               errors += 1
            
            i += 1

        error_rate = errors / len(data)        
        return error_rate

def Pegasos(data, target, devData, devTarget, cParam = 1, _kernel='linear', _degree=1, _coef0=0):

    nCount = len(data[0])
    weightVector = np.zeros(nCount)
    cParam = 1
    lambdaVar = 1 / (nCount * cParam)
    currentTrainingCount = 1
    epochCount = 0
    totalEpoch = 6

    while epochCount <= totalEpoch:

          errors = 0

          startTime = time.time()
          for i in range(0, nCount):
       
              learningRate = 1 / (lambdaVar / currentTrainingCount)

              if (target[i] * np.dot(data[i], weightVector)) < 1:

                  weightVector = weightVector - learningRate * (lambdaVar * weightVector - weightVector * data[i])

              else:
                  weightVector = weightVector - learningRate * lambdaVar * weightVector

              currentTrainingCount += 1

          endTime = time.time()       
          epochCount += 1

          trainError = DevError(weightVector, data, target)
          devError = DevError(weightVector, devData, devTarget)
          
          print("The dev error rate is {:.2%} for epoch = %d", devError, epochCount)

          print("The Training error rate for epoch %d is %f ", epochCount, trainError )
      
          print("The Pegasos ran for %s seconds" % (endTime - startTime))
          #print("The number of support vectors are: ", str(len(clf.support_vectors_)))

    return weightVector

def test(dataSet, data, model, bias, cParam = 1):
    errors = sum(y * (model.dot(vecx) + bias) <= 0 for vecx, y in data)
    error_rate = errors / len(data)

    print("The", dataSet, "error rate is {:.2%} for C = {:.2f}".format(error_rate[0], cParam))


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

          w = Pegasos(train_data, train_target, dev_data, dev_target)
