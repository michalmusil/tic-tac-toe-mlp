# Tic Tac Toe multilayer perceptron
This repository contains a part of school project focused on observing if a neural network can learn to play a simple tic tac toe game without the use of reinforced learning. 

## Dataset generation
The dataset necessary for training the neural network is generated using an already existing implementation of the tic tac toe game (which uses MINIMAX algorithm). 
A JSON file with appropriate game combinations is the output of this process.

## Model training
Using the above mentioned dataset, a model is trained in python using TensorFlow Keras. The trained model is than exported and later used in the second part of the project, 
which is an implementation of tic tac toe in Android, which uses the trained model as an AI adversary. [Android implementation](https://github.com/michalmusil/ticTacToeAndroid)
