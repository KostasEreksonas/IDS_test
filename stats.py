#!/usr/bin/env python3

import NSL_KDD

def distribution(path):
    """Count normal and malicious connections"""
    df = NSL_KDD.dataframe(path)
    normal = df['attack_type'].value_counts()['normal']
    attack = len(df) - normal
    data = {"Normal":normal, "Attack":attack}
    return data

def attack_types(path):
    """Count ocurrences of attack types available in a dataset"""
    df = NSL_KDD.dataframe(path)
    attacks = df['attack_type'].unique()
    attacks = list(attacks)
    attacks.remove('normal')
    data = {}
    for attack in attacks:
        data[attack] = df['attack_type'].value_counts()[attack]
    return data

def attack_subtypes(path):
    """Count subtypes of each attack"""
    df = NSL_KDD.dataframe(path)
    subtypes = df['class'].unique()
    subtypes = list(subtypes)
    subtypes.remove('normal')
    data = {}
    for subtype in subtypes:
        data[subtype] = df['class'].value_counts()[subtype]
    return data

def group_attacks(path):
    """Group attack classes by attack types"""
    df = NSL_KDD.dataframe(path)
    values = df['attack_type'].unique()
    values = list(values)
    values.remove('normal')
    data = {}
    for value in values:
        new_df = df[df['attack_type'] == value]
        data[value] = list(new_df['class'].unique())
    return data

def features(path):
    """Count feature statistics"""
    df = NSL_KDD.dataframe(path)
    values = df[''].unique()
    return values
