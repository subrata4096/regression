#!/usr/bin/python
import numpy as np
from minepy import MINE
import pylab as plt
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from fields import *

#global inputColumnNames
#global measuredColumnNames
#global outputColumnNames

#global inputColumnNameToIndexMapFromFile
#global measuredColumnNameToIndexMapFromFile
#global outputColumnNameToIndexMapFromFile

#mic_score_threshold = 0.00

def print_stats(mine,inputFeatureName,targetName,mic_score_threshold = 0.0):
    print "in=",inputFeatureName, " tareget=", targetName,"\tMIC", mine.mic(), "\tmic threshold: \t" , mic_score_threshold
    #print "MAS", mine.mas()
    #print "MEV", mine.mev()
    #print "MCN (eps=0)", mine.mcn(0)
    #print "MCN (eps=1-MIC)", mine.mcn_general()

def doMICAnalysisOfInputVariables(inArr, targetArr,targetName, mic_score_threshold, targetQualityMap = None):
	#if(targetQuality == None):
	#	return inArr
	#print inArr
	#global inputColumnNameToIndexMapFromFile
        #global measuredColumnNameToIndexMapFromFile
        #global outputColumnNameToIndexMapFromFile

	selected_inArr = []
	selected_inArr_indexes = []
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
		print getGlobalObject("inputColumnNameToIndexMapFromFile")
		inputFeatureName = getGlobalObject("inputColumnNameToIndexMapFromFile")[i]
		print_stats(mine,inputFeatureName,targetName,mic_score_threshold)
		if(targetQualityMap != None):
			targetQualityMap.append(float(mine.mic()))
		#l = list(x)
		#selected_inArr = np.concatenate((selected_inArr, np.array(l)), axis=0)
		#print k
		#print mine.mic()
		if(float(mine.mic()) > mic_score_threshold):
			selected_inArr.append(x)
			selected_inArr_indexes.append(i)
		k = k + 1	
		
	selected_inArr = np.array(selected_inArr).transpose()
	#print selected_inArr
        return selected_inArr, selected_inArr_indexes
	#return selected_inArr
