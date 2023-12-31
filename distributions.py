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

def types(path):
    """Graph a distribution of attack types"""
    dataset_name = path.split("/")[-1].split(".")[0]
    df = stats.attack_types(path)
    keys = list(df.keys())
    values = list(df.values())
    # Plotting the graph
    plots.bar_graph(path, dataset_name, keys, values, "Attack_dist", "Attacks", "Samples", "Attack distribution", "distributions")

def split_dictionary(input_dict, size):
    """Split given dictionary into multpile dictionaries of a given size"""
    res = []
    new_dict = {}
    for key, value in input_dict.items():
        if len(new_dict) < size:
            new_dict[key] = value
        else:
            res.append(new_dict)
            new_dict = {key: value}
    res.append(new_dict)
    return res

def classes(path):
    """Graph a distribution of attack classes"""
    dataset_name = path.split("/")[-1].split(".")[0]
    df = stats.attack_subtypes(path)
    df_sorted = sorted(df.items(), key=lambda x:x[1])
    df = dict(df_sorted)
    data = split_dictionary(df, 4)
    for x in range(len(data)):
        keys = list(data[x].keys())
        values = list(data[x].values())
        # Plotting the graph
        plots.bar_graph(path, dataset_name, keys, values, f"Class_dist-{x+1}", "Classes", "Samples", "Class distribution", "distributions")

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
