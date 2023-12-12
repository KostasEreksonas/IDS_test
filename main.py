#!/usr/bin/env python3

import models
import plots

def main():
    path = "data/NSL_KDD/KDDTrain+.txt"
    plots.bar_graph(path, "protocol_type", "Type", "Occurrences", "Protocol occurences by type")
    plots.bar_graph(path, "attack_type", "Type", "Occurrences", "Attack occurences by type")
    plots.bar_graph(path, "logged_in", "Logged In (1 - Yes, 0 - No)", "Occurrences", "Success")
    #plots.correlation(path)
    #models.neural_network()

if __name__ == "__main__":
    main()
