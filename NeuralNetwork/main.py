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
rawData = network.loadDataFromFile("../Data/trainMoves.json")
#equalizedTrainData = network.equalizeData(rawTrainData)

trainData, testData = network.splitDataToTrainAndTest(dataToSplit=rawData, testRatio=0.15)

train = network.adjustDataForModel(trainData)
test = network.adjustDataForModel(testData)

continuation = True
model = None
while continuation:
    model = network.trainModel(train, test)
    loss, accuracy = model.evaluate(test[0], test[1])
    if loss < 0.49 and accuracy > 0.83:
        continuation = False

#network.saveModel(model, "adam_sparseCE_e10_bs15_relu64_tanh128_tanh64_vs0,1")

"""

loadedModel = network.loadModel("adam_sparseCE_e10_bs15_relu64_tanh128_tanh64_vs0,1")

rawData = network.loadDataFromFile("../Data/testMoves.json")
testData = network.adjustDataForModel(rawData)
loadedModel.evaluate(testData[0],testData[1])

#userTest(loadedModel)

