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
        cols = f.read()
    cols = cols.split()
    columns = []
    for col in cols:
        columns.append(col)
    return columns

# Read attack types
def attacks():
    with open("dataset/attack.types", 'r') as f:
        attacks = f.read()
    attacks = attacks.split()
    attack_name,attack_type = [[] for x in range(2)]
    attack_dict = {}
    for x in range(0,len(attacks)):
        if x == 0 or x % 2 == 0:
            attack_name.append(attacks[x])
        elif x % 2 != 0:
            attack_type.append(attacks[x])
    for x in range(0,len(attack_type)):
        attack_dict[attack_name[x]] = attack_type[x]
    #inverse = {v: k for k, v in attack_dict.items()}
    #attack_dict = inverse
    return attack_dict

# Adding Attack Type column
def add_feature(path, columns, rows):
    df = pd.read_csv(path, names = columns)
    df['Attack Type'] = df['dst_host_srv_rerror_rate'].map(rows)
    return df

def shape(dataframe):
    return dataframe.shape

def find_missing(dataframe):
    return dataframe.isnull().sum()

def main():
    path = "dataset/KDDTrain+_20Percent.txt"
    columns = features()
    rows = attacks()
    #print(add_feature(path, columns, rows))
    df = add_feature(path, columns, rows)
    print(f"Shape: {shape(df)}")
    print(f"Missing:\n{find_missing(df)}")

if __name__ == "__main__":
    main()
