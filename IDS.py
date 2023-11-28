#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
    df['Attack Type'] = df['dst_host_srv_rerror_rate'].map(rows)
    return df

def shape(dataframe):
    return df.shape

def find_missing(df):
    return df.isnull().sum()

def get_coorelation(df):
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    corr = df.corr(numeric_only=True)
    plt.figure(figsize =(15, 12))
    sns.heatmap(corr)
    plt.show()

def main():
    path = "dataset/KDDTrain+_20Percent.txt"
    columns = features()
    rows = attacks()
    df = add_feature(path, columns, rows)
    get_coorelation(df)

if __name__ == "__main__":
    main()
