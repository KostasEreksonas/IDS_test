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
attack_name,attack_type = [[] for x in range(2)]
attack_dict = {}
for x in range(0,len(attacks)):
    if x == 0:
        attack_name.append(attacks[x])
    elif x % 2 == 0:
        attack_name.append(attacks[x])
    elif x % 2 != 0:
        attack_type.append(attacks[x])
print(f"attack_name: {attack_name}, attack_type {attack_type}")
for x in range(0,len(attack_type)):
    attack_dict[attack_name[x]] = attack_type[x]
print(attack_dict)
