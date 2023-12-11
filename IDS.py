#!/usr/bin/env python3

import NSL_KDD
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical, plot_model

def line_graph(data, line1, line2, title, ylabel, xlabel, filename):
    """Plot a line graph for training and testing results"""
    plt.plot(data.history[line1])
    plt.plot(data.history[line2])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.savefig(f"plots/results/{filename}")
    print(f"[+] Graph saved at: plots/results/{filename}")

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
    print("\n[+] Graphs of a created neural network\n------------------------------------------------\n[+] Neural network block scheme")
    plot_model(model, to_file='plots/model/model.png')
    print(f"[+] Graph saved at: plots/model/model.png\n[+] Neural network accuracy graph")
    line_graph(history, "accuracy", "val_accuracy", "Model accuracy", "Accuracy", "Epoch", "accuracy.png")
    print("[+] Neural network loss value graph")
    line_graph(history, "loss", "val_loss", "Model loss", "Loss", "Epoch", "loss.png")

def main():
    neural_network()

if __name__ == "__main__":
    main()
