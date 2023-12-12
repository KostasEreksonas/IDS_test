#!/usr/bin/env python3

import models
import plots

def draw_plots():
    path = ["data/NSL_KDD/KDDTrain+.txt",
            "data/NSL_KDD/KDDTest+.txt",
            "data/NSL_KDD/KDDTrain+_20Percent.txt",
            "data/NSL_KDD/KDDTest-21.txt"]
    for x in path:
        dataset_name = x.split("/")[-1].split(".")[0]
        plots.bar_graph(x, dataset_name, "protocol_type", "Type", "Occurrences", "Protocol occurences by type")
        plots.bar_graph(x, dataset_name, "attack_type", "Type", "Occurrences", "Attack occurences by type")
        plots.bar_graph(x, dataset_name, "logged_in", "Logged In (1 - Yes, 0 - No)", "Occurrences", "Success")
        plots.correlation(dataset_name, x)

def main():
    draw_plots()
    #models.neural_network()

if __name__ == "__main__":
    main()
