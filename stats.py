#!/usr/bin/env python3

import NSL_KDD

#  ---------
# | Numbers |
#  ---------

class Statistics:
    def __init__(self, dataframe, column, data):
        self.dataframe = dataframe
        self.column = column
        self.data = data
    """Get unique column values and count their ocurrences"""
    def stats(self):
        values = self.dataframe[self.column].unique()
        for value in values:
            self.data[value] = self.dataframe[self.column].value_counts()[value]
        return self.data

#  -----------
# | Relations |
#  -----------

class Relations(Statistics):
    def __init__(self, dataframe, column, data, column2):
        Statistics.__init__(self, dataframe, column, data)
        self.column2 = column2
    def group(self):
        """Group a column by a different column"""
        values = self.dataframe[self.column].unique()
        for value in values:
            df = self.dataframe[self.dataframe[self.column] == value]
            self.data[value] = list(df[self.column2].unique())
        return self.data
