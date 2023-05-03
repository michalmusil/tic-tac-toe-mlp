# Author: Michal Musil
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import random
import matplotlib.pyplot as plt


BOARD_JSON_KEY = "board"
NEXT_MOVE_JSON_KEY = "move"
CATEGORIES = [0, 1, 2, 3, 4, 5, 6, 7, 8]

def loadDataFromFile(filePath):
    movesData = []
    with open(filePath, "r") as allData:
        dataList = json.load(allData)
        for data in dataList:
            try:
                board = data[BOARD_JSON_KEY]
                move = CATEGORIES.index(
                    data[NEXT_MOVE_JSON_KEY]
                )
                movesData.append([board, move])
            except Exception as ex:
                print("failed to load dataset instance: ", ex)
    return movesData

def equalizeData(loadedData): 
    # Getting minimum category occurance
    categoryOccurances = [0] * 9
    for board, move in loadedData:
        index = CATEGORIES.index(move)
        categoryOccurances[index] += 1
    
    maxOfEachClass = min(categoryOccurances)
    categoryOccurances = [0] * 9
    equalizedData = []
    for board, move in loadedData:
        occurancesOfThisMove = categoryOccurances[CATEGORIES.index(move)]
        if occurancesOfThisMove < maxOfEachClass:
            equalizedData.append([board, move])
            categoryOccurances[CATEGORIES.index(move)] += 1

    return equalizedData

def splitDataToTrainAndTest(dataToSplit, testRatio):
    if testRatio >= 1 or testRatio <= 0:
        raise Exception("Test ratio must be a number between 0 and 1 excluded")
    random.shuffle(dataToSplit)
    splittingIndex = len(dataToSplit) - round((len(dataToSplit)-1) * testRatio)
    trainData = dataToSplit[0:splittingIndex]
    testData = dataToSplit[splittingIndex:len(dataToSplit)]
    return (trainData, testData)


def adjustDataForModel(loadedData):
    inputs = []
    labels = []
    for input, label in loadedData:
        inputs.append(np.array(input).astype('int32'))
        labels.append(label)
    inputsArray = np.array(inputs)
    labelsArray = np.array(labels)
    return(inputsArray, labelsArray)

def saveModel(model, modelName):
    model.save(f"models/{modelName}.model")

def loadModel(modelName):
    model = tf.keras.models.load_model(f"models/{modelName}.model")
    return model

def exportToTfLite(model, path):
    convert = tf.lite.TFLiteConverter.from_keras_model(model)
    liteModel = convert.convert()
    open(path, "wb").write(liteModel)

def displayTrainHistory(history):
    plt.plot(history.history['accuracy'], label='Přesnost (v %/100)')
    plt.plot(history.history['loss'], label='Odchylka (chyba)')
    plt.title('Vývoj přesnosti a chyby při testování')
    plt.xlabel('Epocha')
    plt.legend(loc="lower left")
    plt.show()




def trainModel(trainData, testData):
    X, Y = trainData
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = keras.activations.relu))
    model.add(keras.layers.Dense(128, activation = keras.activations.tanh))
    model.add(keras.layers.Dense(64, activation = keras.activations.tanh))
    model.add(keras.layers.Dense(9, activation = keras.activations.softmax))

    model.compile(
        optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.001),
        loss = keras.losses.SparseCategoricalCrossentropy(),
        metrics = ['accuracy']
    )

    trainHistory = model.fit(X, Y, epochs = 10, batch_size = 15, validation_split = 0.15)

    print("Evaluation:")
    model.evaluate(testData[0], testData[1])
    return (model, trainHistory)






