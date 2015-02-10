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

global TargetErrorDataMap

#=> convert target list to 2d list
#[1 2 3 4] => [[1] [2] [3] [4]] => beacuse that's what following functions expect
def listTo2DArray(theList):
	list2D = []
	for item in theList:
		list2D.append([item])
	#print list2D
	return np.array(list2D)
	
#takes, input parameter array, target array and target name
def getRegressionFunctionForEachTarget(inArr, tarArr,tname):	
	#fit a polynomial regression of degree 2 using Lasso as underlying linear regression model
	reg = doPolyRegression(inArr, tarArr,tname,2,fitUse="Lasso")
	#print reg
	return reg

#this function will fit the training set for each feature sample for each target and populate the regression function in DS
def populateRegressionFunctionForEachTarget():	
	for targetkey in TargetErrorDataMap.keys():
                tarErrData = TargetErrorDataMap[targetkey]
		for featureKey in tarErrData.FeatureErrorDataMap.keys():
			featureErrData = tarErrData.FeatureErrorDataMap[featureKey]
			trainObs = featureErrData.TrainingObservations
			if(trainObs.observeType == "TEST"):
				#only do fit for "TRAIN" observations. We will use "TEST" to calculate error
				continue
	
			inArr = trainObs.ParamArr
			targetArr = trainObs.TargetArr
			#fit the regression function based on training params and target
			regressFunc = getRegressionFunctionForEachTarget(inArr,targetArr,targetkey)
			#print "here:", targetkey, featureKey, regressFunc
			featureErrData.RegressionFunction = regressFunc
			#print str(tarErrData)
				
def populatePredictionsForTestSamples():
	for targetkey in TargetErrorDataMap.keys():
                tarErrData = TargetErrorDataMap[targetkey]
                for featureIndex in tarErrData.FeatureErrorDataMap.keys():
                        featureErrData = tarErrData.FeatureErrorDataMap[featureIndex]
                        testObsList = featureErrData.TestObservations
                        regressFunc = featureErrData.RegressionFunction
			
			trainSetPoints = featureErrData.TrainingObservations.ParamArr
			meanPoint,StdDev = getMeanAndStandardDevOfTrainingSetOfTrainingSetForAFeature(featureIndex,trainSetPoints)
			
			for testObs in testObsList:
				if(testObs.observeType == "TRAIN"):     
                                	#We will use ONLY "TEST" observations to predict and then calculate error
                                	continue
                        	inArr = testObs.ParamArr
                        	targetArr = testObs.TargetArr
				predicted = regressFunc.predict(inArr)
				testObs.PredictedArr = predicted
				#print "target: ", targetArr, " predicted: ", predicted 
				#NOW CALCULATE ERROR --------------------------------
				#calculate percentage error. Be acreful about devide by zero!!
				error = (targetArr[0] - predicted[0])*1.0/float(targetArr[0]) if (targetArr[0] != 0) else (targetArr[0] - predicted[0])*1.0
				#now also store this error in the same DataStructure. Will be used for distance based profiling
				testObs.PredictionErrArr = error
				#calculate the distance of this target from the center of training set, so that we can have a profile of error varying with distance
				distance = getDistanceOfFeaturesFromTrainingSet(meanPoint,StdDev,featureIndex,inArr)
				testObs.distanceToTargetArr = distance

def populateSamplesInErrorDataStructure(dataFile,inArr,measuredArr,outArr):
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

	#populateSamplesInErrorDataStructure(dataFile,inputDataArr,measuredDataArrT,outputDataArrT)
	populateSamplesInErrorDataStructure(dataFile,inputDataArr,measuredDataArr,outputDataArr)

	
	#populate regression function for each target and for samples sorted based on each feature
	populateRegressionFunctionForEachTarget()
        populatePredictionsForTestSamples()	
	printFullErrorDataStructure()
