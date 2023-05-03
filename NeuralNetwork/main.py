# Author: Michal Musil
import tensorflow as tf
import network
import numpy as np
import time
import random
import matplotlib.pyplot as plt

CURRENT_MODEL_NAME = "adam_sparseCE_e10_bs15_relu64_tanh128_tanh64_vs0,1"
TF_LITE_MODEL_PATH = "models/tflite/ticTacToe.tflite"
DATASET_PATH = "../Data/trainMoves2.json"


def userTest(model):
    while True:
        print("Type in a board state (nine numbers, 1 = currentPlayersMark, 0 = emptyMark, -1 = enemiesMark) like 1,0,0,-1.....:")
        stringInput = input()
        try:
            listInput = [convertToIntList(stringInput)]
            arrayInput = np.array(listInput)

            # Time mesured prediction
            timeBefore = takeCurrentMillis()
            predictions = model.predict([arrayInput])[0]
            timeAfter = takeCurrentMillis()
            duration = timeAfter - timeBefore

            result = np.argmax(predictions)
            print("Result: next move on index: ", result)
            print("Prediction took: ", duration, "milliseconds.")
            print("_________________-")
        except Exception as ex:
            print(ex)

def takeCurrentMillis():
    timeStamp = round(time.time() * 1000)
    return timeStamp

def convertToIntList(stringList):
    intList = []
    stringList = stringList.split(',')
    for s in stringList:
        intList.append(int(s))
    return intList






def trainAndSaveModel():
    rawData = network.loadDataFromFile(DATASET_PATH)
    #equalizedTrainData = network.equalizeData(rawTrainData)

    trainData, testData = network.splitDataToTrainAndTest(dataToSplit = rawData, testRatio = 0.1)

    train = network.adjustDataForModel(trainData)
    test = network.adjustDataForModel(testData)

    model, history = network.trainModel(train, test)
    network.displayTrainHistory(history)

    network.saveModel(model, CURRENT_MODEL_NAME)

def evaluateModelSpeed():
    numberOfTestRuns = 100
    durations = []
    loadedModel = network.loadModel(CURRENT_MODEL_NAME)
    rawData = network.loadDataFromFile(DATASET_PATH)
    #testData = network.adjustDataForModel(rawData)
    #loadedModel.evaluate(testData[0],testData[1])
    for i in range(numberOfTestRuns):
        boardStateIndex = random.randint(0, len(rawData)-1)
        boardStateToTest = np.array([rawData[boardStateIndex][0]])

        timeBefore = takeCurrentMillis()
        predictions = loadedModel.predict([boardStateToTest])[0]
        timeAfter = takeCurrentMillis()

        durations.append(timeAfter - timeBefore)

    plt.plot(durations, label='Doba trvání')
    plt.title('Rychlost modelu')
    plt.xlabel('Pokus')
    plt.ylabel('Milisekundy')
    plt.legend(loc="lower left")
    plt.show()






def loadModelAndSaveToTfLite():
    loadedModel = network.loadModel(CURRENT_MODEL_NAME)
    network.exportToTfLite(model = loadedModel, path = TF_LITE_MODEL_PATH)    

def checkTfLiteModel():
    liteModel = tf.lite.Interpreter(model_path=TF_LITE_MODEL_PATH)
    liteModel.allocate_tensors()
    inputDetails = liteModel.get_input_details()
    outputDetails = liteModel.get_output_details()
    print("Input details: ", inputDetails)
    print("Output details: ", outputDetails)

    adjustedList = np.array([1,0,1,0,1,-1,-1,-1,0]).astype('int32')# expected output is 1
    inputToCheck = [adjustedList]

    liteModel.set_tensor(tensor_index = inputDetails[0]['index'], value = inputToCheck)
    liteModel.invoke()
    result = np.argmax(liteModel.get_tensor(outputDetails[0]['index']))
    print("Result for: ", inputToCheck, " is: ", result)






trainAndSaveModel()
loadModelAndSaveToTfLite()
checkTfLiteModel()
evaluateModelSpeed()
