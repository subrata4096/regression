#!/usr/bin/python -W ignore::DeprecationWarning
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
import sys
import os
from regressFit import *
from micAnalysis import *
from drawPlot import *
from detectAnomaly import *
from fields import *
from pickleDump import *
from analyze import *
from errorDatastructure import *
from errorAnalysis import *
#from errorAnalysisMain import *
from errorAnalysisControlled import *


def createErrorHistogram():
        tgtErrDataMap = getGlobalObject("TargetErrorDataMap")
        for targetkey in tgtErrDataMap.keys():
                tarErrData = tgtErrDataMap[targetkey]
                for featureIndex in tarErrData.FeatureErrorDataMap.keys():
                        featureErrData = tarErrData.FeatureErrorDataMap[featureIndex]
                        testObsList = featureErrData.TestObservations
                        errorSamples = []
                        for testObs in testObsList:
                                if(testObs.observeType == "TRAIN"):
                                        #We will use ONLY "TEST" observations to predict and then calculate error
                                        continue
                                # keep same error samples in a different list in 1D format for histogram/distribution calculations (inefficient!)
                                #print testObs.PredictionErrArr
				errorSamples.append(testObs.PredictionErrArr)
                                tarErrData.errors.append(testObs.PredictionErrArr)
				
			doHistogramPlot(errorSamples,targetkey,featureIndex,False)


if __name__ == "__main__":
	dataFile = sys.argv[1]
	initializeGlobalObjects(dataFile)
	productionDataFile = ""
	print "DataFile: " , dataFile , "\n"        
	print "Input variables", getGlobalObject("inputColumnNames")
	print "Meassured variables", getGlobalObject("measuredColumnNames")
	print "Output variables", getGlobalObject("outputColumnNames")
	
	#these are for figure dump
	#dumpDir = setActiveDumpDirectory(dataFile)	
	#setGlobalObject("activeDumpDirectory",dumpDir)	
	#if(os.path.exists(dumpDir) == False):
        #	os.mkdir(dumpDir)
	
	#get general dump dir
	dumpDir = makeDumpDirectory(dataFile)	
	setGlobalObject("activeDumpDirectory",dumpDir)

	inputDataArr,measuredDataArr,outputDataArr = readInputMeasurementOutput(dataFile)
	
	with warnings.catch_warnings():
    		warnings.simplefilter("ignore")
    		fxn()	
		selectedInputDataArr = selectImportantFeaturesByMICAnalysis(inputDataArr,measuredDataArr,outputDataArr,mic_score_threshold_global)
		populateSamplesInErrorDataStructure(dataFile,selectedInputDataArr,measuredDataArr,outputDataArr,True)

	
	populateRegressionFunctionForEachTarget(2)
	#populateRegressionFunctionForEachTarget(3)
        
	populatePredictionsForTestSamples(True) #pass this is a histogram profile flow	
	

	createErrorHistogram()
	
	#print getGlobalObject("TargetErrorDataMap")["module:measure:PAPI:PAPI_TOT_INS"].errors
	dumpTargetErrMap(getGlobalObject("TargetErrorDataMap"),getGlobalObject("activeDumpDirectory"),dataFile)
	#tMap = loadTargetErrMap(getGlobalObject("activeDumpDirectory"),True,dataFile)
	#print tMap["module:measure:PAPI:PAPI_TOT_INS"].errors

	#prints
	#printFullErrorDataStructure()
	#printErrorDistributionProfileMapForTargetAndFeatureMap()
	#print getGlobalObject("selectedOriginalColIndexMap")
	#dumpSelectedFeaturesMap(getSelectedColumnNames(getGlobalObject("selectedOriginalColIndexMap")),getGlobalObject("activeDumpDirectory"), dataFile)
	#theSelectedDict = loadSelectedFeaturesMap(getGlobalObject("activeDumpDirectory"),True,dataFile)
	#print "loaded ", theSelectedDict

	
