#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def features():
    """Read dataset names"""
    with open("data/NSL_KDD/kddcup.names", 'r') as f:
        cols = f.read().split()
    columns = []
    for col in cols:
        columns.append(col)
    return columns

def attacks():
    """Put attack names and types into a dictionary {name:type}"""
    with open("data/NSL_KDD/attack.types", 'r') as f:
        data = f.read().split()
    attack_name,attack_type = [[] for x in range(2)]
    for x in range(0,len(data)):
        attack_name.append(data[x]) if x == 0 or x % 2 == 0 else attack_type.append(data[x])
    attacks = {}
    for x in range(0,len(attack_type)):
        attacks[attack_name[x]] = attack_type[x]
    return attacks

def add_feature(path, columns, rows):
    """Add attack type feature to a dataframe"""
    df = pd.read_csv(path, names = columns)
    df['Attack Type'] = df['class'].map(rows)
    return df

def dataframe(path):
    columns = features()
    rows = attacks()
    df = add_feature(path, columns, rows)
    return df

def shape(path):
    df = dataframe(path)
    return df.shape

def encode_features(path, columns, rows):
    """Encode text data using one-hot encoding method"""
    df = add_feature(path, columns, rows)
    pmap = {'icmp':0, 'tcp':1, 'udp':2}
    fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10}
    smap = {'aol':0, 'auth':1, 'bgp':2, 'courier':3, 'csnet_ns':4, 'ctf':5, 'daytime':6, 'discard':7, 'domain':8, 'domain_u':9, 'echo':10, 'eco_i':11, 'ecr_i':12, 'efs':13, 'exec':14, 'finger':15, 'ftp':16, 'ftp_data':17, 'gopher':18, 'harvest':19, 'hostnames':20, 'http':21, 'http_2784':22, 'http_443':23, 'http_8001':24, 'imap4':25, 'IRC':26, 'iso_tsap':27, 'klogin':28, 'kshell':29, 'ldap':30, 'link':31, 'login':32, 'mtp':33, 'name':34, 'netbios_dgm':35, 'netbios_ns':36, 'netbios_ssn':37, 'netstat':38, 'nnsp':39, 'nntp':40, 'ntp_u':41, 'other':42, 'pm_dump':43, 'pop_2':44, 'pop_3':45, 'printer':46, 'private':47, 'red_i':48, 'remote_job':49, 'rje':50, 'shell':51, 'smtp':52, 'sql_net':53, 'ssh':54, 'sunrpc':55, 'supdup':56, 'systat':57, 'telnet':58, 'tftp_u':59, 'time':60, 'tim_i':61, 'urh_i':62, 'urp_i':63, 'uucp':64, 'uucp_path':65, 'vmnet':66, 'whois':67, 'X11':68, 'Z39_50':69}
    cmap = {'back':0, 'land':1, 'neptune':2, 'pod':3, 'smurf':4, 'teardrop':5, 'processtable':6, 'udpstorm':7, 'mailbomb':8, 'apache2':9, 'ipsweep':10, 'mscan':11, 'nmap':12, 'portsweep':13, 'saint':14, 'satan':15, 'ftp_write':16, 'guess_passwd':17, 'imap':18, 'multihop':19, 'phf':20, 'warezmaster':21, 'warezclient':22, 'spy':23, 'sendmail':24, 'xlock':25, 'snmpguess':26, 'named':27, 'xsnoop':28, 'snmpgetattack':29, 'worm':30, 'buffer_overflow':31, 'loadmodule':32, 'perl':33, 'rootkit':34, 'xterm':35, 'ps':36, 'httptunnel':37, 'sqlattack':38, 'normal':39}
    tmap = {'normal':0, 'dos':1, 'probe':2, 'r2l':3, 'u2r':3}
    df['protocol_type'] = df['protocol_type'].map(pmap)
    df['flag'] = df['flag'].map(fmap)
    df['service'] = df['service'].map(smap)
    df['class'] = df['class'].map(cmap)
    df['Attack Type'] = df['Attack Type'].map(tmap)
    return df

def preprocessing(path):
    """Preprocess data for training"""
    columns = features()
    rows = attacks()
    scaler = MinMaxScaler()
    data = encode_features(path, columns, rows)
    X = data[data.columns[:43]]
    X = scaler.fit_transform(X)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    Y = data[data.columns[-1]]
    return X, Y
