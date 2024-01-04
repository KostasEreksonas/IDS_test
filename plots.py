#!/usr/bin/env python3

import NSL_KDD
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, plot_model

def bar_graph(dataset_name, keys, values, column_name, xlabel, ylabel, title, folder_name):
    """Draw a bar graph"""
    fig = plt.figure(figsize = (50, 50))
    plt.bar(keys, values, color ='maroon', width = 0.4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(visible=None, which='both', axis='both')
    plt.savefig(f"plots/{folder_name}/{dataset_name}-{column_name}.png")
    print(f"Plot saved to plots/{folder_name}/{dataset_name}-{column_name}.png")

def block_scheme(model, path):
    """Draw a block scheme"""
    plot_model(model, to_file=path)
    print(f"[+] Block scheme saved at: {path}")

def correlation(dataset_name, path):
    """Draw a heatmap"""
    df = NSL_KDD.dataframe(path)
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    corr = df.corr(numeric_only=True)
    plt.figure(figsize =(15, 12))
    sns.heatmap(corr)
    plt.savefig(f"plots/features/{dataset_name}-heatmap.png")
    print(f"Plot saved to plots/features/{dataset_name}-heatmap.png")

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
