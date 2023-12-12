#!/usr/bin/env python3

import models
import plots

def main():
    path = "data/NSL_KDD/KDDTrain+.txt"
    plots.correlation(path)
    #models.neural_network()

if __name__ == "__main__":
    main()
