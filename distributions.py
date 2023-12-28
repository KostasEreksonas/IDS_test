#!/usr/bin/env python3

import plots
from scipy.stats import binom
import matplotlib.pyplot as plt
import NSL_KDD

def bernoulli():
    # Count total values and normal values
    path = "data/NSL_KDD/KDDTrain+.txt"
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
    pass
