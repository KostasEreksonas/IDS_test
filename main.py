#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import preprocessing
import models
import plots
import distributions
import stats

path = ["data/NSL_KDD/KDDTrain+.txt",
        "data/NSL_KDD/KDDTest+.txt",
        "data/NSL_KDD/KDDTrain+_20Percent.txt",
        "data/NSL_KDD/KDDTest-21.txt"]
def statistics():
    columns = ["protocol_type", "service", "flag", "class", "attack_type"]
    for x in path:
        name = x.split("/")[-1].split(".")[0]
        for column in columns:
            stat = stats.Statistics(preprocessing.dataframe(x), column, {})
            col = distributions.Plot(name, stat.stats(), f"{column}_dist", f"{column}", "Samples", f"{column} distribution", "distributions")
            col.graph()

def algorithms():
    models.reccurent_neural_network()

def main():
    algorithms

if __name__ == "__main__":
    main()
