#!/usr/bin/python
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

#global inputColumnNames
#global measuredColumnNames
#global outputColumnNames
#global regressionDict

#global activeDumpDirectory

#global TargetErrorDataMap
#global ErrorDistributionProfileMapForTargetAndFeature

#=> convert target list to 2d list
#[1 2 3 4] => [[1] [2] [3] [4]] => beacuse that's what following functions expect
def listTo2DArray(theList):
	list2D = []
	for item in theList:
		list2D.append([item])
	#print list2D
	return np.array(list2D)
	

#this function will fit the training set for each feature sample for each target and populate the regression function in DS
def populateRegressionFunctionForEachTarget(degree):
	tgtErrDataMap = getGlobalObject("TargetErrorDataMap")	
	for targetkey in tgtErrDataMap.keys():
                tarErrData = tgtErrDataMap[targetkey]
		for featureKey in tarErrData.FeatureErrorDataMap.keys():
			featureErrData = tarErrData.FeatureErrorDataMap[featureKey]
			trainObs = featureErrData.TrainingObservations
			if(trainObs.observeType == "TEST"):
				#only do fit for "TRAIN" observations. We will use "TEST" to calculate error
				continue
	
			inArr = trainObs.ParamArr
			targetArr = trainObs.TargetArr
			#fit the regression function based on training params and target
			regressFunc = getRegressionFunctionForEachTarget(inArr,targetArr,targetkey,degree)
			#print "here:", targetkey, featureKey, regressFunc
			featureErrData.RegressionFunction = regressFunc
			#print str(tarErrData)



def populateErrorProfileFunctions():
	tgtErrDataMap = getGlobalObject("TargetErrorDataMap")	
	for targetkey in tgtErrDataMap.keys():
                tarErrData = tgtErrDataMap[targetkey]
                for featureIndex in tarErrData.FeatureErrorDataMap.keys():
                        featureErrData = tarErrData.FeatureErrorDataMap[featureIndex]
                        testObsList = featureErrData.TestObservations
			distanceList = []
			errorList = []
			errorSamples = []
                        for testObs in testObsList:
                                if(testObs.observeType == "TRAIN"):
                                        #We will use ONLY "TEST" observations to predict and then calculate error
                                        continue
				#make a 2d list of distance and errors => to be used for curve fitting
                                distanceList.append([testObs.DistanceToTargetArr])
                                errorList.append(testObs.PredictionErrArr)
				# also keep same error samples in a different list in 1D format for histogram/distribution calculations (inefficient!)
                                errorSamples.append(testObs.PredictionErrArr)
			
			#featureName = getInputParameterNameFromColumnIndex(featureIndex)
			featureName = getInputParameterNameFromFeatureIndex(featureIndex)
			errDistProfile = curveFitErrorSamplesWithDistance(targetkey,featureName,distanceList,errorList,errorSamples)	
			#we already have errDistProfile populated in the map. So skip everything below	
			#if targetkey in ErrorDistributionProfileMapForTargetAndFeature.keys():
			#	ErrorDistributionProfileMapForTargetAndFeature[targetkey][featureName] = errDistProfile
			#else:
			#	featureMap = {}
			#	featureMap[featureName] = errDistProfile
			#	ErrorDistributionProfileMapForTargetAndFeature[targetkey] = featureMap
	
				
def populatePredictionsForTestSamples(forHistogramAnalysis):
	tgtErrDataMap = getGlobalObject("TargetErrorDataMap")	
	errDistProfMapForTargetAndFeature = getGlobalObject("ErrorDistributionProfileMapForTargetAndFeature")
	for targetkey in tgtErrDataMap.keys():
                tarErrData = tgtErrDataMap[targetkey]
                for featureIndex in tarErrData.FeatureErrorDataMap.keys():
                        featureErrData = tarErrData.FeatureErrorDataMap[featureIndex]
                        testObsList = featureErrData.TestObservations
                        regressFunc = featureErrData.RegressionFunction
			
			trainSetPoints = featureErrData.TrainingObservations.ParamArr
			if(forHistogramAnalysis == False): #if it is not a histogram profile flow
				meanPoint,StdDev = getMeanAndStandardDevOfTrainingSetOfTrainingSetForAFeature(featureIndex,trainSetPoints)
				featureName = getInputParameterNameFromFeatureIndex(featureIndex)
				errDistProfile = errorDistributionProfile(featureName,targetkey)
				errDistProfile.MeanPointOfTrainingSet = meanPoint
				errDistProfile.StandardDeviationOfTrainingSet = StdDev

				#Put this errorProfile object into the data structure (that is this 2 level map)
				#This will be populated later with error regression function
				#start populating data structure
				if targetkey in errDistProfMapForTargetAndFeature.keys():
                                	errDistProfMapForTargetAndFeature[targetkey][featureName] = errDistProfile
                        	else:
                                	featureMap = {}
                                	featureMap[featureName] = errDistProfile
                                	errDistProfMapForTargetAndFeature[targetkey] = featureMap
				#end populating data structure

			#end NOT "forHistogramAnalysis" flow

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

				if(forHistogramAnalysis == False): #if it is not a histogram profile flow

					#calculate the distance of this target from the center of training set, so that we can have a profile of error varying with distance
					distance = getDistanceOfFeaturesFromTrainingSet(meanPoint,StdDev,featureIndex,inArr)
					testObs.DistanceToTargetArr = distance
				#end NOT "forHistogramAnalysis" flow
def getSelectedRowKey(featureIndex,numFeatures,row):
	keyStr = ""
	for i in range(numFeatures):
		if(i==featureIndex):
			continue
		keyStr = keyStr + ":" + str(row[i])
	return keyStr
		
def findControlledSubsetReadings(featureIndex,numFeatures,sortedArr):
	i = 0
	#print featureIndex, numFeatures
	rowKey = "1-1-1"
	inDict = {}
	selectedArr = []
	for row in sortedArr:
                rowKey = getSelectedRowKey(featureIndex,numFeatures,row)
                if rowKey in inDict.keys():
			#print "rowKey", rowKey, "  row " , row 
                        selectedArr.append(row)
                else:
			if(i==0):
				#print "rowKey", rowKey, "  row " , row 
                        	inDict[rowKey] = True
                        	selectedArr.append(row)
         	i = i + 1

	selectedArr = np.array(selectedArr)
	return selectedArr


# returns data structure TargetErrorData => data structure per target for train and test samples
def generateControlledTrainingAndTestSetsForDistanceProfilingForEachTarget(inArr,targetArr,targetName):
        numSamples = inArr.shape[0]
        numFeatures = inArr.shape[1]
        mergedArr = getMergedInputAndTargetArray(inArr,targetArr)

        numTotalColumns = mergedArr.shape[1]

        #featureErrorDataList = []
        targetErrData = TargetErrorData(targetName)
        for featureIndex in range(numFeatures):
                sortedArr = getSortedArrayBasedOnColumn(mergedArr,featureIndex)
		#sortedArr = findControlledSubsetReadings(featureIndex,numFeatures,sortedArr)
		#print str(sortedArr)
		#np.savetxt("a.txt",sortedArr, delimiter='\t')
                #Each train and test sample is a MERGED array of input and target.
                #we need to split these later

                # to start with (according to document error_profiling.pdf) we will start with just use one resample
                #trainAndTestSamples = getSamplesFromBootstrap(sortedArr,1,0.66,True)

                #do not use bootstrap resampling. according to document error_profiling.pdf), use first few from
                #sorted list of params, as training set and far points as test set
                #trainAndTestSamples = getSamplesFromSortedParams(sortedArr,1,0.9,False)  # 90% used for training
                trainAndTestSamples = getSamplesFromSortedParams(sortedArr,1,0.7,False)  # 70% used for training 
                #trainAndTestSamples = getSamplesFromSortedParams(sortedArr,1,0.5,False)   # 50% used for training
                #trainAndTestSamples = getSamplesFromSortedParams(sortedArr,1,0.3,False)  # 30% used for training
                #trainAndTestSamples = getSamplesFromSortedParams(sortedArr,1,0.1,False)  # 10% used for training
                #trainAndTestSamples = getSamplesFromSortedParams(sortedArr,1,0.2,False)
                #trainAndTestSamples = getSamplesFromSortedParams(sortedArr,1,0.66,False)

                feErr = FeatureErrorData()
		feErr.name = featureIndex
                for key in trainAndTestSamples.keys():
                        trainSample,testSamples = trainAndTestSamples[key]
                        traingObs = getObservationsFromMergedSamples(trainSample,numFeatures, numTotalColumns,"TRAIN")
                        feErr.TrainingObservations = traingObs
                        for testSample in testSamples:
                                testObs = getObservationsFromMergedSamples(testSample,numFeatures, numTotalColumns,"TEST")
                                feErr.TestObservations.append(testObs)

                #now append this featureErrorData into the list. This is for a target, 
                #So latter should be attached to the target name
                #featureErrorDataList.append(feErr)  
                targetErrData.FeatureErrorDataMap[feErr.name] = feErr

        #return featureErrorDataList                 
        return targetErrData


def populateSamplesInErrorDataStructure(dataFile,inArr,measuredArr,outArr,useBootStrap):
	tgtErrDataMap = getGlobalObject("TargetErrorDataMap")	
	i = 0
	for targetArr in measuredArr:
		t = getGlobalObject("measuredColumnNames")[i]
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

		#This is for controlled experiments
		if(useBootStrap):
			targetErrData = generateControlledTrainingAndTestSetsForErrorHistogramForEachTarget(inArr,targetArrT,t)
		else:
			targetErrData = generateControlledTrainingAndTestSetsForDistanceProfilingForEachTarget(inArr,targetArrT,t)
		
		tgtErrDataMap[t] = targetErrData
		i = i + 1

	i = 0
	for targetArr in outArr:
		t = getGlobalObject("outputColumnNames")[i]
		#reg = doFitForTarget(inArr,targetArr,t)
		#regressionDict[t] = reg
		#targetArrT = map(lambda t: list(t), targetArr)
		targetArrT = listTo2DArray(targetArr) #=> convert target list to 2d list
		if(useBootStrap):
			targetErrData = generateControlledTrainingAndTestSetsForErrorHistogramForEachTarget(inArr,targetArrT,t)
                else:  
			targetErrData = generateControlledTrainingAndTestSetsForDistanceProfilingForEachTarget(inArr,targetArrT,t)
		
		tgtErrDataMap[t] = targetErrData
		i = i + 1
			

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
	#global inputColumnNameToIndexMapFromFile
        #global measuredColumnNameToIndexMapFromFile
        #global outputColumnNameToIndexMapFromFile
	#print "here 1", getGlobalObject("inputColumnNameToIndexMapFromFile")
	selectedInputDataArr = selectImportantFeaturesByMICAnalysis(inputDataArr,measuredDataArr,outputDataArr,mic_score_threshold_global)
	#selectedInputDataArr = selectImportantFeaturesByMICAnalysis(inputDataArr,measuredDataArr,outputDataArr,0.0)

	#measuredDataArrT = map(lambda t: list(t), measuredDataArr)
	#outputDataArrT = map(lambda t: list(t), outputDataArr)
	#print "measuredArrT :", measuredDataArrT

	#populateSamplesInErrorDataStructure(dataFile,inputDataArr,measuredDataArrT,outputDataArrT)
	populateSamplesInErrorDataStructure(dataFile,selectedInputDataArr,measuredDataArr,outputDataArr,False)

	
	#populate regression function for each target and for samples sorted based on each feature
	#print "here 2", getGlobalObject("inputColumnNameToIndexMapFromFile")
	populateRegressionFunctionForEachTarget(3)
	#print "here 3", getGlobalObject("inputColumnNameToIndexMapFromFile")
        populatePredictionsForTestSamples(False)	
        print getGlobalObject("columnIndexToInArrIndexMap")
        print getGlobalObject("inputColumnIndexToNameMapFromFile")
	populateErrorProfileFunctions()

	#prints
	#printFullErrorDataStructure()
	#printErrorDistributionProfileMapForTargetAndFeatureMap()
	#print getGlobalObject("selectedOriginalColIndexMap")
	#dumpSelectedFeaturesMap(getSelectedColumnNames(getGlobalObject("selectedOriginalColIndexMap")),getGlobalObject("activeDumpDirectory"), dataFile)
	#theSelectedDict = loadSelectedFeaturesMap(getGlobalObject("activeDumpDirectory"),True,dataFile)
	#print "loaded ", theSelectedDict

	#picklepath,cPicklepath = dumpErrorDistributionProfileMap(getGlobalObject("ErrorDistributionProfileMapForTargetAndFeature"),getGlobalObject("activeDumpDirectory"),dataFile)
	#errProfMap = loadErrorDistributionProfileMap(getGlobalObject("activeDumpDirectory"),True,dataFile)
	#printErrorDistributionProfileMapForTargetAndFeatureMap(errProfMap)
	
