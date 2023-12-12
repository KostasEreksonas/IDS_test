#!/usr/bin/env python3

import NSL_KDD
import plots
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from tensorflow.keras.optimizers import SGD

def neural_network():
    """A deep neural network model"""
    # Load data
    trainX, trainY = NSL_KDD.preprocessing("data/NSL_KDD/KDDTrain+.txt")
    testX, testY = NSL_KDD.preprocessing("data/NSL_KDD/KDDTest+.txt")
    # Creating model
    model = Sequential()
    model.add(GRU(256, input_shape=(1,43)))     # Gated Recurrent Unit
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))     # Hidden layer #1
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))     # Hidden layer #2
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))   # Output layer
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    start = time.perf_counter()
    print("\n[+] Model training and validation\n---------------------------------------")
    history = model.fit(trainX, trainY, epochs=20, batch_size=16, validation_split=0.1, verbose=1)
    print("\n[+] Model testing\n------------------------")
    score = model.evaluate(testX, testY, batch_size=16)
    end = time.perf_counter()
    print(f"\n[+] Model accuracy\n-----------------------\nAccuracy: {score[1]*100:.2f}%")
    print(f"\n[+] Running time\n-----------------------\nTime: {end-start:.2f} seconds")
    plots.block_scheme(model, "plots/model/model.png")
    plots.line_graph(history, "accuracy", "val_accuracy", "Model accuracy", "Accuracy", "Epoch", "accuracy.png")
    plots.line_graph(history, "loss", "val_loss", "Model loss", "Loss", "Epoch", "loss.png")
