import numpy as np

## Test current Perceptron on developer data and report
## the error rate.

def DevEvaluator(weightVector, devDataArray):

    numberDevData = len(devDataArray)
    devWrong = 0
##    positiveCounter = 0

    for i in range(0, numberDevData):

        if devDataArray[i, -1] == 1:
            y = 1

        else:
            y = -1

        xi = devDataArray[i, 0:-1]

##        if np.dot(xi, weightVector) > 0:
##            positiveCounter += 1

        if y*np.dot(xi, weightVector) <= 0:
            
            devWrong += 1

    devError = (devWrong / len(devDataArray)) * 100
##    positivePercent = (positiveCounter / len(devDataArray)) * 100
    return devError
