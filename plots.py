#!/usr/bin/env python3

import matplotlib.pyplot as plt

def line_graph(data, line1, line2, title, ylabel, xlabel, filename):
    """Plot a line graph for training and testing results"""
    plt.plot(data.history[line1])
    plt.plot(data.history[line2])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.savefig(f"plots/results/{filename}")
    print(f"[+] Graph saved at: plots/results/{filename}")
