#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import NSL_KDD
import models
import plots
import regressions
import distributions
import stats

def draw_plots():
    df = NSL_KDD.dataframe(path)
    data = {}
    keys = df[column_name].unique()
    for key in keys:
        data[key] = df[column_name].value_counts()[key]
    values = list(data.values())
    path = ["data/NSL_KDD/KDDTrain+.txt",
            "data/NSL_KDD/KDDTest+.txt",
            "data/NSL_KDD/KDDTrain+_20Percent.txt",
            "data/NSL_KDD/KDDTest-21.txt"]
    for x in path:
        dataset_name = x.split("/")[-1].split(".")[0]
        plots.bar_graph(x, dataset_name, "protocol_type", "Protocol", "Occurrences", "Protocol occurences by type", "features")
        plots.bar_graph(x, dataset_name, keys, values, "service", "Service", "Occurrences", "Service occurences by type", "features")
        plots.bar_graph(x, dataset_name, keys, values, "flag", "Flag", "Occurrences", "Flag occurences by type", "features")
        plots.bar_graph(x, dataset_name, keys, values, "class", "Class", "Occurrences", "Class occurences by type", "features")
        plots.bar_graph(x, dataset_name, keys, values,"attack_type", "Attack Type", "Occurrences", "Attack occurences by type", "features")
        plots.bar_graph(x, dataset_name, keys, values, "logged_in", "Logged In (1 - Yes, 0 - No)", "Occurrences", "Success", "features")
        plots.correlation(dataset_name, x)

def main():
    #distributions.bernoulli()
    #distributions.multinomial()
    #print(stats.distribution("data/NSL_KDD/KDDTrain+.txt"))
    #print(stats.attack_types("data/NSL_KDD/KDDTrain+.txt"))
    #print(stats.attack_subtypes("data/NSL_KDD/KDDTrain+.txt"))
    #print(stats.group_attacks("data/NSL_KDD/KDDTrain+.txt"))
    distributions.attack("data/NSL_KDD/KDDTrain+.txt")
    #models.reccurent_neural_network()

if __name__ == "__main__":
    main()
