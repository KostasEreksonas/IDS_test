#!/usr/bin/env python3

import stats
import NSL_KDD

def read():
    """Read statistics of a dataset"""
    path = ["data/NSL_KDD/KDDTrain+.txt",
            "data/NSL_KDD/KDDTest+.txt",
            "data/NSL_KDD/KDDTrain+_20Percent.txt",
            "data/NSL_KDD/KDDTest-21.txt"]
    parameters = ["protocol_type", "service", "flag", "class", "attack_type"]
    results,result = [[] for x in range(2)]
    for x in path:
        result.append(x)
        for parameter in parameters:
            p1 = stats.Statistics(NSL_KDD.dataframe(x), parameter, {})
            result.append(parameter)
            result.append(p1.stats())
        results.append(result)
        result = []
    return results

def main():
    results = read()
    for result in results:
        print(result)

if __name__ == '__main__':
    main()
