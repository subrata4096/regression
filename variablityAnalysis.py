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
from analyze import *

global inputColumnNames
global measuredColumnNames
global outputColumnNames
global regressionDict

targetDataQuality = {}

def identifyGoodTargets(targetDataQuality):
	print "\n ==================== Identify good targets ==================\n"
	for key in targetDataQuality.keys():
		#print key
		som = targetDataQuality[key]["som"]
		if(som > 0.3):
			print key, ":- too much variation"
			continue
		
		hasGoodMicScore = False
		miclist = targetDataQuality[key]["mic"]
		for m in miclist:
			if(m < 0.85):
				continue
			else:
				hasGoodMicScore = True
		if(hasGoodMicScore):
			print key, ":- good target"
		else:
			print key, ":- not good mic score"
			
	return

if __name__ == "__main__":
        dataFile = sys.argv[1]
        print "DataFile: " , dataFile , "\n"
        print "Input variables", inputColumnNames
        print "Meassured variables", measuredColumnNames
        print "Output variables", outputColumnNames

        inputDataArr = readDataFile(dataFile,'input')
        
	inputDataArr = np.transpose(inputDataArr)
        
	measuredDataArr = readDataFile(dataFile,'measured')
        outputDataArr = readDataFile(dataFile,'output')

        measureVari = calculateVariability(inputDataArr,measuredDataArr)
        outVari = calculateVariability(inputDataArr,outputDataArr)
	
	i = 0	
	for targetArr in measuredDataArr:
                t = measuredColumnNames[i]
		print "\n Target :  ", t
		print "Avg - Std over mean: ", measureVari[i]
		targetDataQuality[t] = {"som" : measureVari[i], "mic" : []}
		i = i + 1
	i = 0
	for targetArr in outputDataArr:
                t = outputColumnNames[i]
                print "\n Target :  ", t
                print "Avg - Std over mean: ", outVari[i]
		targetDataQuality[t] = {"som" : outVari[i], "mic" : []}

                i = i + 1


	print " ======================== MIC analysis ============================"
	i = 0	
	for targetArr in measuredDataArr:
                t = measuredColumnNames[i]
                print "\n Target :  ", t
        	selected_inArr,selected_inArr_indexs = doMICAnalysisOfInputVariables(inputDataArr, targetArr,0.0, targetDataQuality[t]["mic"]) 
                #print targetDataQuality[t]["mic"]
		i = i + 1
	
	i = 0	
	for targetArr in outputDataArr:
                t = outputColumnNames[i]
		print "\n Target :  ", t
                selected_inArr = doMICAnalysisOfInputVariables(inputDataArr, targetArr, 0.0,targetDataQuality[t]["mic"])
                i = i + 1

	#print targetDataQuality
	identifyGoodTargets(targetDataQuality)
