#!/usr/bin/env python3

import NSL_KDD
import models

def main():
    path = "data/NSL_KDD/KDDTrain+.txt"
    df = NSL_KDD.dataframe(path)
    shape = NSL_KDD.shape(path)
    is_null = NSL_KDD.isnull(path)
    print(f"Dataframe:\n{df}\nShape: {shape}\nIs null:\n{is_null}")
    models.neural_network()

if __name__ == "__main__":
    main()
