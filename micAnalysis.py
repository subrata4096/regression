#!/usr/bin/python
import numpy as np
from minepy import MINE
import pylab as plt
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D

def print_stats(mine):
    print "MIC", mine.mic()
    #print "MAS", mine.mas()
    #print "MEV", mine.mev()
    #print "MCN (eps=0)", mine.mcn(0)
    #print "MCN (eps=1-MIC)", mine.mcn_general()

def doMICAnalysisOfInputVariables(inArr, targetArr):
	#print inArr
	numOfColumns = inArr.shape[1]
	for i in range(numOfColumns):
		x = inArr[:,i]
		x_scaled = preprocessing.scale(x)
		print "x: ", x_scaled
		#print "targetArr: ", targetArr 
		mine = MINE(alpha=0.6, c=15)
		mine.compute_score(x_scaled, targetArr)
		print_stats(mine)
