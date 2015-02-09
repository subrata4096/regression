#!/usr/bin/python
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
import sys
from regressFit import *
from micAnalysis import *
from drawPlot import *
from detectAnomaly import *
from fields import *
from pickleDump import *
from analyze import *
from errorDatastructure import *
from errorAnalysis import *

global inputColumnNames
global measuredColumnNames
global outputColumnNames
global regressionDict

#=> convert target list to 2d list
#[1 2 3 4] => [[1] [2] [3] [4]] => beacuse that's what following functions expect
def listTo2DArray(theList):
	list2D = []
	for item in theList:
		list2D.append([item])
	#print list2D
	return np.array(list2D)
		
def do_error_analysis(dataFile,inArr,measuredArr,outArr):
	i = 0
	for targetArr in measuredArr:
		t = measuredColumnNames[i]
		#reg = doFitForTarget(inArr,targetArr,t)
		#regressionDict[t] = reg
		#fname = dumpModel(dataFile,t,reg)
                #regLoad = loadModel(fname)
		#regressionDict[t] = regLoad
		# returns data structure TargetErrorData
		#print "targetArr :", targetArr
		targetArrT = listTo2DArray(targetArr) #=> convert target list to 2d list
		#targetArrT = map(lambda t: list(t), targetArr)
		#print "targetArrT :", targetArrT
		targetErrData = generateTrainingAndTestSetsForDistanceProfilingForEachTarget(inArr,targetArrT,t)
		TargetErrorDataMap[t] = targetErrData
		i = i + 1

	i = 0
	for targetArr in outArr:
		t = outputColumnNames[i]
		#reg = doFitForTarget(inArr,targetArr,t)
		#regressionDict[t] = reg
		#targetArrT = map(lambda t: list(t), targetArr)
		targetArrT = listTo2DArray(targetArr) #=> convert target list to 2d list 
		targetErrData = generateTrainingAndTestSetsForDistanceProfilingForEachTarget(inArr,targetArrT,t)
		TargetErrorDataMap[t] = targetErrData
		i = i + 1
			

if __name__ == "__main__":
	dataFile = sys.argv[1]
	productionDataFile = ""
	print "DataFile: " , dataFile , "\n"        
	print "Input variables", inputColumnNames
	print "Meassured variables", measuredColumnNames
	print "Output variables", outputColumnNames

 	inputDataArr = readDataFile(dataFile,'input')
	inputDataArr = np.transpose(inputDataArr)
 	measuredDataArr = readDataFile(dataFile,'measured')
	#print "measured"
	#print measuredDataArr
 	outputDataArr = readDataFile(dataFile,'output')

	#measuredDataArrT = map(lambda t: list(t), measuredDataArr)
	#outputDataArrT = map(lambda t: list(t), outputDataArr)
	#print "measuredArrT :", measuredDataArrT

	#do_error_analysis(dataFile,inputDataArr,measuredDataArrT,outputDataArrT)
	do_error_analysis(dataFile,inputDataArr,measuredDataArr,outputDataArr)
	printFullErrorDataStructure()
