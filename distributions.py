#!/usr/bin/env python3

import plots
import stats
from scipy.stats import binom
from scipy.stats import multinomial
import matplotlib.pyplot as plt
import NSL_KDD

def attack(path):
    """Graph a distribution between normal and malicious connections"""
    dataset_name = path.split("/")[-1].split(".")[0]
    df = stats.distribution(path)
    keys = list(df.keys())
    values = list(df.values())
    # Plotting the graph
    plots.bar_graph(path, dataset_name, keys, values, "Connection_dist", "Connections", "Samples", "Attack distribution", "distributions")

def bernoulli(path):
    # Count total values and normal values
    dataset_name = path.split("/")[-1].split(".")[0]
    normal = 0
    df = NSL_KDD.dataframe(path)
    total = len(df.values)
    attack_types = df['attack_type'].values
    for attack_type in attack_types:
        if attack_type == 'normal':
            normal += 1
    # Set parameters for Bernoulli trial
    n = total
    p = normal/total
    # Defining list of r values
    r_values = list(range(n + 1))
    # Obtaining the mean and variance
    mean, var = binom.stats(n, p)
    # List of pmf values
    dist = [binom.pmf(r, n, p) for r in r_values]
    # Printing mean and variance
    print(f"Mean = {str(mean)}")
    print(f"Variance = {str(var)}")
    # Plotting the graph
    plots.bar_graph(path, dataset_name, r_values, dist, "Bernoulli", "Data samples", "Probability", "Bernoulli distribution", "distributions")

def poisson():
    pass

def normal():
    pass

def multinomial():
    path = "data/NSL_KDD/KDDTrain+.txt"
    dataset_name = path.split("/")[-1].split(".")[0]
    normal = 0
    df = NSL_KDD.dataframe(path)
    total = len(df.values)
    unique = df['attack_type'].unique()
    p1, p2, p3, p4, p5 = [0 for x in range(5)]
    attack_types = df['attack_type'].values
    for attack_type in attack_types:
        if attack_type == 'normal':
            p1 += 1
        elif attack_type == 'dos':
            p2 += 1
        elif attack_type == 'r2l':
            p3 += 1
        elif attack_type == 'probe':
            p4 += 1
        elif attack_type == 'u2r':
            p5 += 1
    #print(f"Normal: {p1}, dos: {p2}, r2l: {p3}, probe: {p4}, u2r: {p5}")
    stats = multinomial.pmf(x=[4, 5, 1, 3, 2], n=total, p=[p1/total, p2/total, p3/total, p4/total, p5/total])
