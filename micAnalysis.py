#!/usr/bin/python
import numpy as np
from minepy import MINE
from sklearn import preprocessing
from fields import *

#global inputColumnNames
#global measuredColumnNames
#global outputColumnNames

#global inputColumnNameToIndexMapFromFile
#global measuredColumnNameToIndexMapFromFile
#global outputColumnNameToIndexMapFromFile

mic_score_threshold_global = 0.7

def print_stats(mine,inputFeatureName,targetName,mic_score_threshold = 0.0):
    print "in=",inputFeatureName, " tareget=", targetName,"\tMIC", mine.mic(), "\tmic threshold: \t" , mic_score_threshold
    #print "MAS", mine.mas()
    #print "MEV", mine.mev()
    #print "MCN (eps=0)", mine.mcn(0)
    #print "MCN (eps=1-MIC)", mine.mcn_general()

def chooseIndependantInputVariables(inArr):
	#print inArr
	selected_input_indexes = []
	for i in range(inArr.shape[1]):
		doSelect = True
		for j in range(i):
			if(i == j):
				return
			x = inArr[:,i]
			y = inArr[:,j]
			inputFeatureName1 = getInputParameterNameFromColumnIndex(i)
			inputFeatureName2 = getInputParameterNameFromColumnIndex(j)
                	#print "x: ", x
                	x_scaled = preprocessing.scale(x)
                	y_scaled = preprocessing.scale(y)
                	#print "x: ", x_scaled
                	#print "targetArr: ", targetArr 
                	mine = MINE(alpha=0.6, c=15)
                	mine.compute_score(x_scaled, y_scaled)
			print "Correlation between ",inputFeatureName1,inputFeatureName2, " is ", mine.mic()  
			if(float(mine.mic()) >= 0.9):
				doSelect = False
				print "\n ***** ==> will NOT select ", inputFeatureName1, " as it correlates with ", inputFeatureName2, "\n" 
		#end for
		if(doSelect):
			selected_input_indexes.append(i)

	return selected_input_indexes

#def doMICAnalysisOfTargetVariables(inArr,fullTargetArr, mic_score_threshold):
#for targetArr in fullTargetArr:
		
def doMICAnalysisOfInputVariables(inArr, targetArr,targetName, mic_score_threshold,input_indexes_uncorrelated_features,targetQualityMap = None):
	#if(targetQuality == None):
	#	return inArr
	#print inArr
	#global inputColumnNameToIndexMapFromFile
        #global measuredColumnNameToIndexMapFromFile
        #global outputColumnNameToIndexMapFromFile

 	#print "\n\n\n doMICAnalysisOfInputVariables called \n\n"

	selected_inArr = []
	selected_inArr_indexes = []
	selected_originalColumn_indexes = []

	inColMap = getGlobalObject("inputColumnIndexToNameMapFromFile") #keys are col index and vals are names
	#selected_inArr.append([])
	#print "doMICAnalysisOfInputVariables: ", "inArr.shape: ", inArr.shape
	#print "doMICAnalysisOfInputVariables: ", "targetArr.shape: ", targetArr.shape

	numOfFeatures = 0
	try:
		#(rows,numOfFeatures) = inArr.shape
		numOfFeatures = inArr.shape[1]
	except:
		print "ERROR: \n", inArr
		exit(0)
	k = 0	
	for featureIndex in range(numOfFeatures):
	#for i in inColMap.keys():
		#x = inArr[:,i]
		#x = inArr[:,k]
		# we will choose only uncorrelated features as input
		if(featureIndex not in input_indexes_uncorrelated_features):
			continue

		x = inArr[:,featureIndex]
		#print "x: ", x
		x_scaled = preprocessing.scale(x)
		#print "x: ", x_scaled
		#print "targetArr: ", targetArr 
		mine = MINE(alpha=0.6, c=15)
		mine.compute_score(x_scaled, targetArr)
		#print getGlobalObject("inputColumnNameToIndexMapFromFile")
		#inputFeatureName = getGlobalObject("inputColumnNameToIndexMapFromFile")[i]
		#inputFeatureName = inColMap[i]
		#inputFeatureName = getInputParameterNameFromFeatureIndex(featureIndex)
		inputFeatureName = getInputParameterNameFromColumnIndex(featureIndex)
		print_stats(mine,inputFeatureName,targetName,mic_score_threshold)
		if(targetQualityMap != None):
			targetQualityMap.append(float(mine.mic()))
		#l = list(x)
		#selected_inArr = np.concatenate((selected_inArr, np.array(l)), axis=0)
		#print k
		#print mine.mic()
		if(float(mine.mic()) >= mic_score_threshold):
			selected_inArr.append(x) #keep the input data column
			selected_inArr_indexes.append(k) #keep the index corresponding to that column
			colIdx = getColumnIndexFromFeatureIndex(featureIndex)
			selected_originalColumn_indexes.append(colIdx) #keep the original column index corresponding to that column
			print "----------------- selected: ", inputFeatureName, colIdx, k
			k = k + 1	
		
	selected_inArr = np.array(selected_inArr).transpose()
	#print "\n **** selected: ==== \n", selected_inArr, selected_inArr_indexes,selected_originalColumn_indexes
        return selected_inArr, selected_inArr_indexes, selected_originalColumn_indexes
	#return selected_inArr
