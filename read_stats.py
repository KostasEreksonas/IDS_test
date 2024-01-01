#!/usr/bin/env python3

import stats
import NSL_KDD

def main():
    """Read statistics of a dataset"""
    data = {}
    path = ["data/NSL_KDD/KDDTrain+.txt",
            "data/NSL_KDD/KDDTest+.txt",
            "data/NSL_KDD/KDDTrain+_20Percent.txt",
            "data/NSL_KDD/KDDTest-21.txt"]
    columns = ["protocol_type", "service", "flag", "class", "attack_type"]
    results,name = [[] for x in range(2)]
    for x in path:
        name.append(x.split("/")[-1].split(".")[0])
        for column in columns:
            col = stats.Statistics(NSL_KDD.dataframe(x), column, {})
            data[column] = col.stats()
        results.append(data)
    for x in range(len(name)):
        print(f"{name[x]}:\n{results[x]}")
    var = stats.Relations(NSL_KDD.dataframe(path[0]), "attack_type", {}, "class")
    print(var.group())

if __name__ == '__main__':
    main()
