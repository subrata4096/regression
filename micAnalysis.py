#!/usr/bin/python
import numpy as np
from minepy import MINE
import pylab as plt
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from fields import *

global inputColumnNames
global measuredColumnNames
global outputColumnNames

mic_score_threshold = 0.70

def print_stats(mine,i):
    print inputColumnNames[i], "\tMIC", mine.mic(), "\tmic threshold: \t" , mic_score_threshold
    #print "MAS", mine.mas()
    #print "MEV", mine.mev()
    #print "MCN (eps=0)", mine.mcn(0)
    #print "MCN (eps=1-MIC)", mine.mcn_general()

def doMICAnalysisOfInputVariables(inArr, targetArr):
	#print inArr
	selected_inArr = []
	#selected_inArr.append([])
	numOfColumns = inArr.shape[1]
	k = 0	
	for i in range(numOfColumns):
		x = inArr[:,i]
		#print "x: ", x
		x_scaled = preprocessing.scale(x)
		#print "x: ", x_scaled
		#print "targetArr: ", targetArr 
		mine = MINE(alpha=0.6, c=15)
		mine.compute_score(x_scaled, targetArr)
		print_stats(mine,i)
		#l = list(x)
		#selected_inArr = np.concatenate((selected_inArr, np.array(l)), axis=0)
		#print k
		#print mine.mic()
		if(float(mine.mic()) > mic_score_threshold):
			selected_inArr.append(x)
		k = k + 1	
		
	selected_inArr = np.array(selected_inArr).transpose()
	#print selected_inArr
	return selected_inArr
