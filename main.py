#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import NSL_KDD
import models
import plots
import regressions
import distributions
import stats

def statistics():
    path = ["data/NSL_KDD/KDDTrain+.txt",
            "data/NSL_KDD/KDDTest+.txt",
            "data/NSL_KDD/KDDTrain+_20Percent.txt",
            "data/NSL_KDD/KDDTest-21.txt"]
    name = path[0].split("/")[-1].split(".")[0]
    df = NSL_KDD.dataframe(path[0])
    connection_stats = stats.Statistics(df, "attack_type", {})
    connection = distributions.Plot(path[0], name, connection_stats.stats(), "Connection_dist", "Connections", "Samples", "Connection distribution", "distributions")
    attack.graph()

    plots.bar_graph(path, dataset_name, keys, values, "Attack_dist", "Attacks", "Samples", "Attack distribution", "distributions")
    plots.bar_graph(path, dataset_name, keys, values, f"Class_dist-{x+1}", "Classes", "Samples", "Class distribution", "distributions")

def algorithms():
    models.reccurent_neural_network()

def main():
    statistics()

if __name__ == "__main__":
    main()
