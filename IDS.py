#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from tensorflow.keras.optimizers import SGD

# Read feature names
def features():
    with open("dataset/kddcup.names", 'r') as f:
        cols = f.read().split()
    columns = []
    for col in cols:
        columns.append(col)
    return columns

# Read attack types
def attacks():
    with open("dataset/attack.types", 'r') as f:
        data = f.read().split()
    attack_name,attack_type = [[] for x in range(2)]
    attacks = {}
    for x in range(0,len(data)):
        if x == 0 or x % 2 == 0:
            attack_name.append(data[x])
        elif x % 2 != 0:
            attack_type.append(data[x])
    for x in range(0,len(attack_type)):
        attacks[attack_name[x]] = attack_type[x]
    return attacks

# Adding Attack Type column
def add_feature(path, columns, rows):
    df = pd.read_csv(path, names = columns)
    df['Attack Type'] = df['class'].map(rows)
    return df

def shape(dataframe):
    return df.shape

def find_missing(df):
    return df.isnull().sum()

def get_correlation(df):
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    corr = df.corr(numeric_only=True)
    plt.figure(figsize =(15, 12))
    sns.heatmap(corr)
    plt.show()

# Create a bar plot from dictionary
def bar_graph(data, xlabel, ylabel, title, filename):
    keys = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize = (10, 5))
    plt.bar(keys, values, color ='maroon', width = 0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(visible=None, which='both', axis='both')
    plt.savefig(f"plots/{filename}")
    print(f"Plot saved to plots/{filename}")

def plot(path, columns, rows, column_name, xlabel, ylabel, title, filename):
    df = add_feature(path, columns, rows)
    data = {}
    keys = df[column_name].unique()
    for key in keys:
        data[key] = df[column_name].value_counts()[key]
    bar_graph(data, xlabel, ylabel, title, filename)

def draw_plots(path, columns, rows):
    plot(path, columns, rows, "protocol_type", "Type", "Occurrences", "Protocol occurrences by type", "protocols.png")
    plot(path, columns, rows, "Attack Type", "Type", "Occurrences", "Attack occurrences by type", "attacks.png")
    plot(path, columns, rows, "logged_in", "Logged in (1 - Yes, 0 - No)", "Occurrences", "Successfully logged in", "logged.png")

# Encode text data using one-hot encoding method
def encode_features(path, columns, rows):
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
    train_data = encode_features(path[1], columns, rows)
    test_data = encode_features(path[3], columns, rows)
    trainX = train_data[train_data.columns[:43]]
    trainX = scaler.fit_transform(trainX)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    trainY = train_data[train_data.columns[-1]]
    testX = test_data[test_data.columns[:43]]
    testX = scaler.fit_transform(testX)
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    testY = test_data[test_data.columns[-1]]
    return trainX, trainY, testX, testY

def model(path, columns, rows):
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
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])
    print("\n[+] Model training and validation")
    print("---------------------------------------")
    history = model.fit(trainX, trainY,
              epochs=20,
              batch_size=16, validation_split=0.1, verbose=1)
    print("\n[+] Model testing")
    print("------------------------")
    score = model.evaluate(testX, testY, batch_size=16)
    print("\n[+] Model accuracy")
    print("-----------------------")
    print(f"Accuracy: {score[1]*100}%")

def main():
    path = ["dataset/KDDTrain+.txt", "dataset/KDDTrain+_20Percent.txt", "dataset/KDDTest+.txt", "dataset/KDDTest-21.txt"]
    columns = features()
    rows = attacks()
    model(path, columns, rows)

if __name__ == "__main__":
    main()
