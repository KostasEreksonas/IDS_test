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
    path = ["data/NSL_KDD/KDDTrain+.txt",
            "data/NSL_KDD/KDDTest+.txt",
            "data/NSL_KDD/KDDTrain+_20Percent.txt",
            "data/NSL_KDD/KDDTest-21.txt"]
    for x in path:
        distributions.attack(x)
        distributions.types(x)
        distributions.classes(x)

def main():
    #draw_plots()
    print(NSL_KDD.dataframe("data/NSL_KDD/KDDTrain+.txt"))
    #print(stats.service("data/NSL_KDD/KDDTrain+.txt"))
    #distributions.bernoulli()
    #distributions.multinomial()
    #models.reccurent_neural_network()

if __name__ == "__main__":
    main()
