#!/usr/bin/env python3

import NSL_KDD
import models
import plots

def draw_plots():
    path = ["data/NSL_KDD/KDDTrain+.txt",
            "data/NSL_KDD/KDDTest+.txt",
            "data/NSL_KDD/KDDTrain+_20Percent.txt",
            "data/NSL_KDD/KDDTest-21.txt"]
    for x in path:
        dataset_name = x.split("/")[-1].split(".")[0]
        plots.bar_graph(x, dataset_name, "protocol_type", "Protocol", "Occurrences", "Protocol occurences by type")
        plots.bar_graph(x, dataset_name, "service", "Service", "Occurrences", "Service occurences by type")
        plots.bar_graph(x, dataset_name, "flag", "Flag", "Occurrences", "Flag occurences by type")
        plots.bar_graph(x, dataset_name, "class", "Class", "Occurrences", "Class occurences by type")
        plots.bar_graph(x, dataset_name, "attack_type", "Attack Type", "Occurrences", "Attack occurences by type")
        plots.bar_graph(x, dataset_name, "logged_in", "Logged In (1 - Yes, 0 - No)", "Occurrences", "Success")
        plots.correlation(dataset_name, x)

def main():
    models.neural_network()

if __name__ == "__main__":
    main()
