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
		TargetErrorData = generateTrainingAndTestSetsForDistanceProfilingForEachTarget(inArr,targetArr,t)
		i = i + 1

	i = 0
	for targetArr in outArr:
		t = outputColumnNames[i]
		#reg = doFitForTarget(inArr,targetArr,t)
		#regressionDict[t] = reg
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

	do_error_analysis(dataFile,inputDataArr,measuredDataArr,outputDataArr)

