import tensorflow as tf
import numpy as np
import json


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

def adjustDataForModel(loadedData):
    inputs = []
    labels = []
    for input, label in loadedData:
        inputs.append(np.array(input))
        labels.append(label)
    inputsArray = np.array(inputs)
    labelsArray = np.array(labels)
    return(inputsArray, labelsArray)

def trainModel(trainData, testData):
    X, Y = trainData
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(9, activation=tf.nn.softmax))

    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    model.fit(X, Y, epochs = 10, batch_size = 15)
    
    print("Evaluation:")
    model.evaluate(testData[0], testData[1])
    return model

def saveModel(model, modelName):
    model.save(f"{modelName}.model")

def loadModel(modelName):
    model = tf.keras.models.load_model(f"{modelName}.model")
    return model




