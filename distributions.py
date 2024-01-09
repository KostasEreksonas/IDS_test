#!/usr/bin/env python3

import plots
import stats
from scipy.stats import binom
from scipy.stats import multinomial
import matplotlib.pyplot as plt
import preprocessing

class Plot:
    def __init__(self, name, dataframe, column, xlabel, ylabel, title, folder):
        self.name = name
        self.dataframe = dataframe
        self.column = column
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.folder = folder
    """Graph selected statistics of the dataset"""
    def graph(self):
        keys = list(self.dataframe.keys())
        values = list(self.dataframe.values())
        plots.bar_graph(self.name, keys, values, self.column, self.xlabel, self.ylabel, self.title, self.folder)
