#!/usr/bin/env python3

def features():
    """Read dataset names"""
    with open("data/NSL-KDD/kddcup.names", 'r') as f:
        cols = f.read().split()
    columns = []
    for col in cols:
        columns.append(col)
    return columns

def attacks():
    """Put attack names and types into a dictionary {name:type}"""
    with open("data/NSL-KDD/attack.types", 'r') as f:
        data = f.read().split()
    attack_name,attack_type = [[] for x in range(2)]
    for x in range(0,len(data)):
        attack_name.append(data[x]) if x == 0 or x % 2 == 0 else attack_type.append(data[x])
    attacks = {}
    for x in range(0,len(attack_type)):
        attacks[attack_name[x]] = attack_type[x]
    return attacks
