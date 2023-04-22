import network
import numpy as np

def userTest(model):
    while True:
        print("Type in a board state (nine numbers, 1 = currentPlayersMark, 0 = emptyMark, -1 = enemiesMark) like 1,0,0,-1.....:")
        stringInput = input()
        try:
            listInput = [convertToIntList(stringInput)]
            arrayInput = np.array(listInput)
            predictions = model.predict([arrayInput])[0]
            result = np.argmax(predictions)
            print("Next move on index: ", result)
            print("")
        except Exception as ex:
            print(ex)

def convertToIntList(stringList):
    intList = []
    stringList = stringList.split(',')
    for s in stringList:
        intList.append(int(s))
    return intList


"""
rawTrainData = network.loadDataFromFile("../Data/trainMoves.json")
rawTestData = network.loadDataFromFile("../Data/testMoves.json")

#equalizedTrainData = equalizeData(rawTrainData)

trainData = network.adjustDataForModel(rawTrainData)
testData = network.adjustDataForModel(rawTestData)

model = network.trainModel(trainData, testData)
network.saveModel(model, "adam_sparseCE_e10_bs15_relu64_relu128_relu64")
"""
loadedModel = network.loadModel("adam_sparseCE_e10_bs15_relu64_relu128_relu64")

userTest(loadedModel)