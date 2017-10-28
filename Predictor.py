import numpy as np

def Predictor(model, testData):
    with open("income-data/income.test.txt", "r") as f:
        content = f.readlines()
        output = open('income-data/income.test.predicted.txt', 'w')
        positiveResults = 0

        prediction = model.predict(testData)
        
        for i in range(len(content)):
            
            if prediction[i] == -1:
                # <= 50
                output.write(content[i].replace('\n',' <=50\n'))
            else:
                # >50
                output.write(content[i].replace('\n',' >50\n'))
                positiveResults += 1
        output.close()
        
    return positiveResults
