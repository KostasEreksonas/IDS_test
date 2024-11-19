#!/bin/sh

ip="205.174.165.80"

wget http://${ip}/CICDataset/CIC-UNSW/CICFlowMeter_out.csv
wget http://${ip}/CICDataset/CIC-UNSW/Data.csv
wget http://${ip}/CICDataset/CIC-UNSW/Label.csv
