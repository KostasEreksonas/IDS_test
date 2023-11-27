#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Read feature names
with open("dataset/kddcup.names", 'r') as f:
    cols = f.read()
cols = cols.split()
columns = []
for col in cols:
    columns.append(col)

print(f"Features: {columns}\nLength: {len(columns)}")

# Read attack types
with open("dataset/attack.types", 'r') as f:
    attacks = f.read()

attacks = attacks.split()
arr1,arr2 = [[] for x in range(2)]
attack_dict = {}
for x in range(0,len(attacks)):
    if x == 0:
        arr1.append(attacks[x])
    elif x % 2 == 0:
        arr1.append(attacks[x])
    elif x % 2 != 0:
        arr2.append(attacks[x])
print(f"arr1: {arr1}, arr2 {arr2}")
for x in range(0,len(arr2)):
    attack_dict[arr1[x]] = arr2[x]
print(attack_dict)
