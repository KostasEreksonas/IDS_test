#!/usr/bin/env python3

import NSL_KDD

#  ---------
# | Numbers |
#  ---------

class Statistics:
    """Get unique column values and count their ocurrences"""
    def __init__(self, dataframe, column, data):
        self.dataframe = dataframe
        self.column = column
        self.data = data
    def stats(self):
        values = self.dataframe[self.column].unique()
        for value in values:
            self.data[value] = self.dataframe[self.column].value_counts()[value]
        return self.data

#  -----------
# | Relations |
#  -----------

def group_attacks(path):
    """Group attack classes by attack types"""
    df = NSL_KDD.dataframe(path)
    values = df['attack_type'].unique()
    data = {}
    for value in values:
        new_df = df[df['attack_type'] == value]
        data[value] = list(new_df['class'].unique())
    return data
