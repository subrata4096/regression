#!/usr/bin/python

import os

nodeSizeList = [5000,10000,20000,30000,40000,50000,60000,70000,80000]
densityList = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]

for numNodes in nodeSizeList:
	for density in densityList:
		command = "./erdos_renyi.py " + str(numNodes) + " " + str(density)
		os.system(command)
