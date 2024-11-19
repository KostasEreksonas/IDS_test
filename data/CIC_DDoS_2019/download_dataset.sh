#!/bin/sh

ip="205.174.165.80"

wget http://${ip}/CICDataset/CICDDoS2019/Dataset/PCAPs/
wget http://${ip}/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-01-12.zip && 7z x CSV-01-12.zip
wget http://${ip}/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-03-11.zip && 7z x CSV-03-11.zip
