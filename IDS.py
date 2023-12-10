#!/usr/bin/env python3

import imports/data/NSL_KDD/NSL_KDD
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical, plot_model

def add_feature(path, columns, rows):
    """Add attack type feature to a dataframe"""
    df = pd.read_csv(path, names = columns)
    df['Attack Type'] = df['class'].map(rows)
    return df

def shape(df):
    """Get shape of a given dataframe"""
    return df.shape

def find_missing(df):
    """Find missing values"""
    return df.isnull().sum()

def get_correlation(df):
    """Draw a feature heatmap"""
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    corr = df.corr(numeric_only=True)
    plt.figure(figsize =(15, 12))
    sns.heatmap(corr)
    plt.savefig(f"plots/features/heatmap.png")
    print(f"Plot saved to plots/features/heatmap.png")

def bar_graph(data, xlabel, ylabel, title, filename):
    """Draw a bar graph"""
    keys = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize = (10, 5))
    plt.bar(keys, values, color ='maroon', width = 0.4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(visible=None, which='both', axis='both')
    plt.savefig(f"plots/features/{filename}")
    print(f"Plot saved to plots/features/{filename}")

def plot(df, graph, column_name, xlabel, ylabel, title, filename):
    """Plot a graph"""
    data = {}
    keys = df[column_name].unique()
    for key in keys:
        data[key] = df[column_name].value_counts()[key]
    if graph == "Bar":
        bar_graph(data, title, xlabel, ylabel, filename)

def draw_plots(path, columns, rows):
    """Draw bar plots for visualizing features"""
    df = add_feature(path, columns, rows)
    plot(df, "Bar", "protocol_type", "Type", "Occurrences", "Protocol occurrences by type", "protocols.png")
    plot(df, "Bar", "Attack Type", "Type", "Occurrences", "Attack occurrences by type", "attacks.png")
    plot(df, "Bar", "logged_in", "Logged in (1 - Yes, 0 - No)", "Occurrences", "Successfully logged in", "logged.png")

def encode_features(path, columns, rows):
    """Encode text data using one-hot encoding method"""
    df = add_feature(path, columns, rows)
    pmap = {'icmp':0, 'tcp':1, 'udp':2}
    fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10}
    smap = {'aol':0, 'auth':1, 'bgp':2, 'courier':3, 'csnet_ns':4, 'ctf':5, 'daytime':6, 'discard':7, 'domain':8, 'domain_u':9, 'echo':10, 'eco_i':11, 'ecr_i':12, 'efs':13, 'exec':14, 'finger':15, 'ftp':16, 'ftp_data':17, 'gopher':18, 'harvest':19, 'hostnames':20, 'http':21, 'http_2784':22, 'http_443':23, 'http_8001':24, 'imap4':25, 'IRC':26, 'iso_tsap':27, 'klogin':28, 'kshell':29, 'ldap':30, 'link':31, 'login':32, 'mtp':33, 'name':34, 'netbios_dgm':35, 'netbios_ns':36, 'netbios_ssn':37, 'netstat':38, 'nnsp':39, 'nntp':40, 'ntp_u':41, 'other':42, 'pm_dump':43, 'pop_2':44, 'pop_3':45, 'printer':46, 'private':47, 'red_i':48, 'remote_job':49, 'rje':50, 'shell':51, 'smtp':52, 'sql_net':53, 'ssh':54, 'sunrpc':55, 'supdup':56, 'systat':57, 'telnet':58, 'tftp_u':59, 'time':60, 'tim_i':61, 'urh_i':62, 'urp_i':63, 'uucp':64, 'uucp_path':65, 'vmnet':66, 'whois':67, 'X11':68, 'Z39_50':69}
    cmap = {'back':0, 'land':1, 'neptune':2, 'pod':3, 'smurf':4, 'teardrop':5, 'processtable':6, 'udpstorm':7, 'mailbomb':8, 'apache2':9, 'ipsweep':10, 'mscan':11, 'nmap':12, 'portsweep':13, 'saint':14, 'satan':15, 'ftp_write':16, 'guess_passwd':17, 'imap':18, 'multihop':19, 'phf':20, 'warezmaster':21, 'warezclient':22, 'spy':23, 'sendmail':24, 'xlock':25, 'snmpguess':26, 'named':27, 'xsnoop':28, 'snmpgetattack':29, 'worm':30, 'buffer_overflow':31, 'loadmodule':32, 'perl':33, 'rootkit':34, 'xterm':35, 'ps':36, 'httptunnel':37, 'sqlattack':38, 'normal':39}
    tmap = {'normal':0, 'dos':1, 'probe':2, 'r2l':3, 'u2r':3}
    df['protocol_type'] = df['protocol_type'].map(pmap)
    df['flag'] = df['flag'].map(fmap)
    df['service'] = df['service'].map(smap)
    df['class'] = df['class'].map(cmap)
    df['Attack Type'] = df['Attack Type'].map(tmap)
    return df

def data_preprocessing(path, columns, rows, scaler):
    """Preprocess data for training and testing ML models"""
    # Get data
    train_data = encode_features(path[0], columns, rows)
    test_data = encode_features(path[2], columns, rows)
    # Prepare train data
    trainX = train_data[train_data.columns[:43]]
    trainX = scaler.fit_transform(trainX)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # Prepare test data
    testX = test_data[test_data.columns[:43]]
    testX = scaler.transform(testX)
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # Get labels
    trainY = train_data[train_data.columns[-1]]
    testY = test_data[test_data.columns[-1]]
    return trainX, trainY, testX, testY

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

def gaussianNB(path, columns, rows):
    """Gaussian Naive Bayes implementation"""
    trainX, trainY, testX, testY = data_preprocessing(path, columns, rows, MinMaxScaler())
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    clfg = GaussianNB()
    start_time = time.time()
    clfg.fit(trainX, trainY.values.ravel())
    end_time = time.time()
    print("Training time: ", end_time-start_time)
    start_time = time.time()
    y_test_pred = clfg.predict(trainX)
    end_time = time.time()
    print("Testing time: ", end_time-start_time)
    print("Train score is:", clfg.score(trainX, trainY))
    print("Test score is:", clfg.score(testX, testY))

def neural_network(path, columns, rows):
    """A deep neural network model"""
    trainX, trainY, testX, testY = data_preprocessing(path, columns, rows, MinMaxScaler())
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

def to_csv(path, columns, rows):
    add_feature(path[0], columns, rows).to_csv('Train.csv')
    add_feature(path[1], columns, rows).to_csv('Train20.csv')
    add_feature(path[2], columns, rows).to_csv('Test.csv')
    add_feature(path[3], columns, rows).to_csv('Test21.csv')

def main():
    path = ["impots/data/NSL_KDD/KDDTrain+.txt", "imports/data/NSL_KDD/KDDTrain+_20Percent.txt", "imports/data/NSL_KDD/KDDTest+.txt", "imports/data/NSL_KDD/KDDTest-21.txt"]
    columns = NSL_KDD.features()
    rows = NSL_KDD.attacks()
    neural_network(path, columns, rows)
    #df = add_feature(path[0], columns, rows)
    #print(df)
    #print(df['protocol_type'].value_counts().to_frame())
    #print(df['service'].value_counts().to_frame())
    #print(df['class'].value_counts().to_frame())

if __name__ == "__main__":
    main()
