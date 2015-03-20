#!/usr/bin/python -W ignore::DeprecationWarning
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import LeavePOut
from sklearn.metrics.pairwise import *
import scipy
from scipy.spatial.distance import *
from scipy.stats.stats import pearsonr
from minepy import MINE
from sklearn import preprocessing 
import numpy as np
from micAnalysis import *

import sys
import math
import operator
from regressFit import *
from micAnalysis import *
from drawPlot import *
from detectAnomaly import *
from fields import *
from analyze import *
from pickleDump import *
from errorDatastructure import *

global inputColumnNames
global measuredColumnNames
global outputColumnNames
global regressionDict

def getSortedTupleFromDictionary(theDict):
	sorted_tuple = sorted(theDict.items(), key=operator.itemgetter(1))
	return sorted_tuple

#merge two arrays side by side (horizontally)
def getMergedInputAndTargetArray(inArr, targetArr):
	#print inArr
	#print targetArr
	mergedArr = np.concatenate((inArr, targetArr), axis=1)
	#print mergedArr
	return mergedArr

#sort an array based on the column index provided
def getSortedArrayBasedOnColumn(inArr,columnIndex):
	sortedArr = inArr[inArr[:,columnIndex].argsort()]
	#print "Sorted array, based on column = ", columnIndex
	#print sortedArr
	return  sortedArr

def calculateCorrelationBetweenVectors(x,y):
	#x = scipy.array([-0.65499887,  2.34644428, 3.0])
 	#y = scipy.array([-1.46049758,  3.86537321, 21.0])
	#The Pearson correlation coefficient measures the linear relationship between two datasets. 
	#Strictly speaking, Pearson correlation requires that each dataset be normally distributed. 
	#correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation. 
	#Correlations of -1 or +1 imply an exact linear relationship. 

	#The p-value roughly indicates the probability of an uncorrelated system producing datasets that have a Pearson correlation at least as extreme as the one computed from these datasets. 
	#The p-values are not entirely reliable but are probably reasonable for datasets larger than 500 or so.
	#print "X = " , x, "\nY = ", y
	#corr, p_value = pearsonr(x, y)
	commonSize = 0
	if(len(x) < len(y)):
		commonSize = len(x)
	else:
		commonSize = len(y)
	x_sorted = np.sort(x)
	y_sorted = np.sort(y)
	
	x_sorted = x_sorted[ : (commonSize - 1)]
	y_sorted = y_sorted[ : (commonSize - 1)]
	
	x_scaled = preprocessing.scale(x_sorted)
	y_scaled = preprocessing.scale(y_sorted)

	mine = MINE(alpha=0.6, c=15)
        mine.compute_score(x_scaled, y_scaled)	
	corr = float(mine.mic())
	#return 
	#print "correlation :", corr
	return corr

#print the error data structure map
def printErrorDataStructureMap(errDSList):
	for errDS in errDSList:
		print "Training obs: \n\t" + str(errDS.TrainingObservations)
		testObs = errDS.TestObservations
		for o in testObs:
			print "Test obs: \n\t" + str(o)
		

def getSamplesFromBootstrap(inArr, resampleNumber, percentageInTrainSet, useBootstrap):
	numSamples = inArr.shape[0]
	bs = cross_validation.Bootstrap(numSamples, resampleNumber, percentageInTrainSet, random_state=0)
	#print bs
	#print inArr
        testTrainPairMap = {}
        bootStrapIndex = 0;
        for train_index, test_index in bs:
                bootStrapIndex = bootStrapIndex + 1
                trainList = []
                testDataPoints = []
                #print("TRAIN:", train_index, "TEST:", test_index)
                for idx in train_index:
                        #print idx
                        trainList.append(inArr[idx])
                trainArr = np.array(trainList)
                
		for idx in test_index:
                        testDataPoints.append(inArr[idx])
		testArr = np.array(testDataPoints)

                testTrainPairMap[bootStrapIndex] = (trainArr,testDataPoints)

        #print testTrainPairMap
        #for key in testTrainPairMap.keys():
                #print "Key = ", key, "  value = ", testTrainPairMap[key]

        return testTrainPairMap


#takes the merged and sorted array where datapoints and targets are side by side
#create samples through bootstrap resampling(with replacements)
#returns a map of tupples indexed by bootstap index (number of times resampling was done)
#each tuple has a merged array for training and a merged list of points for testing
#resampleNumber => number of times we would resample 
#percentageInTrainSet => what percentage of the samples will be used as train set (rest as test)
def getSamplesFromSortedParams(inArr, resampleNumber, percentageInTrainSet, useBootstrap):
	numSamples = inArr.shape[0]
	testTrainPairMap = {}
	bootStrapIndex = 0;
	if(useBootstrap):
		#print "\n\n== ERROR == ! Please call getSamplesFromBootstrap instead..\n\n"
		#print "\n\n == ARE YOU SURE? YOU ARE CALLING BOOTSTRAP FUNCTION FOR HISTOGRAM!! .. \n\n"
		testTrainPairMap = getSamplesFromBootstrap(inArr,resampleNumber,percentageInTrainSet,useBootstrap)
	else:
		trainingSize = (int)(0.5 + (float)(numSamples * percentageInTrainSet))   #0.5 for rounding up of sample size
		trainSet = [x for x in range(trainingSize)]
		testSet = [x for x in range(trainingSize,numSamples)]
		#print trainingSize, trainSet, testSet
		bootStrapIndex = bootStrapIndex + 1
		trainList = []
		testDataPoints = []
  		#print("TRAIN:", trainSet, "TEST:", testSet)
		for idx in trainSet:
			#print idx
			trainList.append(inArr[idx])
		trainArr = np.array(trainList)
		for idx in testSet:
			testDataPoints.append(inArr[idx])
		
		testArr = np.array(testDataPoints)
		testTrainPairMap[bootStrapIndex] = (trainArr,testArr)
			
	#print testTrainPairMap
	#for key in testTrainPairMap.keys():
	#	print "Key = ", key, "  value = ", testTrainPairMap[key] 

	return testTrainPairMap

def getStandardizedValues(inArr):

	return

def getMeanOfObservations(inArr):
	#inArr = np.array([[0, 0], [1,11], [2,12],[3,13]])  => format
	#print inArr
        #output:   [ 1.5  9. ]

	centerOfRef = np.mean(inArr,axis=0)
	#print centerOfRef
	return centerOfRef

def getPairWiseDistance(refArr, otherArr):
	#distArr = pairwise_distances(refArr, otherArr, metric='mahalanobis')
	#distArr = pairwise_distances(refArr, otherArr, metric='euclidean')
	distArr = pairwise_distances(refArr, otherArr, metric='seuclidean')
	#distArr = pairwise_distances(refArr, otherArr, metric='jaccard')
	print distArr
        m = np.mean(distArr)
	print "mean  m = ", m
	return distArr

#Greg: you didnot specify the distance metric we will use. I think it was (distance to center of training set)/(standard deviation of training set).
def getMeanAndStandardDevOfTrainingSetOfTrainingSetForAFeature(featureIndex,trainArr):
	#get the feature/param column corresponding to this feature index
	featureColumn = trainArr[:,featureIndex]
	meanPoint = np.mean(featureColumn)
	StdDev = np.std(featureColumn)	
	#print "##########################---------------> Training feature, meanpt, stddev", featureColumn, meanPoint, StdDev
	return meanPoint,StdDev

#Greg: you didnot specify the distance metric we will use. I think it was (distance to center of training set)/(standard deviation of training set).
def getDistanceAlongOneFeature(meanPoint,stdDev,testPoint):
	distance = 0.0
        if(stdDev != 0):
                diff = abs(float(testPoint - meanPoint))
                #distance = diff
                distance = diff/stdDev
        else:
                distance = abs(float(testPoint - meanPoint))
        #print "##########################---------------> TestPoint, Distance", testPoint, distance
        return distance

def getDistanceOfFeaturesFromTrainingSet(meanPoint,stdDev,featureIndex,testParamArr):
	#get the feature/param corresponding to this feature index
	testPoint = testParamArr[featureIndex]
	distance = getDistanceAlongOneFeature(meanPoint,stdDev,testPoint)
	return distance

def getStandardizedEuclideanDistance(refArr, otherArr):

	#standardized Euclidean distance, meaning that it is the Euclidean distance calculated on standardized data. 

	#For efficiency reasons, the euclidean distance between a pair of row vector x and y is computed as:
	#dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

	distArr = euclidean_distances(refArr, otherArr)
	print distArr
	return distArr	

def getObservationsFromMergedSamples(samples,numOfFeatures,totalNumberOfColumns,observationType):
	#print "merged sample: " , samples
        #while splitting horizontally, we should give [numOfFeatures, totalNumberOfColumns] argumnet to hsplit
	#which will return 3 arrays:
	# 1. samples[:numOfFeatures] => essntially the data-point part of the merged array 
	# 2. samples[numOfFeatures : totalNumberOfColumns] => essentially the target array
	# 3. samples[totalNumberOfColumns:] => essentially an empty array -> we are not interested
	
	#print "Num features: ",numOfFeatures, " total columns: " , totalNumberOfColumns
	splittedList = np.hsplit(samples, np.array([numOfFeatures, totalNumberOfColumns] ))
	paramArr = splittedList[0]  #or inArr or data point array
	targetArr = splittedList[1]
	#pass observationType as "TRAIN" or "TEST"
	obs = Observations(paramArr,targetArr,observationType)
	return obs

def generateTrainingAndTestSetsForErrorHistogramForEachTarget(inArr,targetArr,targetName):
	numSamples = inArr.shape[0]
        numFeatures = inArr.shape[1]
        mergedArr = getMergedInputAndTargetArray(inArr,targetArr)
        numTotalColumns = mergedArr.shape[1]
        targetErrData = TargetErrorData(targetName)
	trainAndTestSamples = getSamplesFromSortedParams(mergedArr,100,0.9,True) #use bootstrap
        feErr = FeatureErrorData()
        feErr.name = targetName
        for key in trainAndTestSamples.keys():
        	trainSample,testSamples = trainAndTestSamples[key]
        	traingObs = getObservationsFromMergedSamples(trainSample,numFeatures, numTotalColumns,"TRAIN")
        	feErr.TrainingObservations = traingObs
        	for testSample in testSamples:
        		testObs = getObservationsFromMergedSamples(testSample,numFeatures, numTotalColumns,"TEST")
                	feErr.TestObservations.append(testObs)
	
	targetErrData.FeatureErrorDataMap[feErr.name] = feErr
        return targetErrData
	
# returns data structure TargetErrorData => data structure per target for train and test samples
def generateTrainingAndTestSetsForDistanceProfilingForEachTarget(inArr,targetArr,targetName):
	numSamples = inArr.shape[0]
	numFeatures = inArr.shape[1]
	mergedArr = getMergedInputAndTargetArray(inArr,targetArr)

        numTotalColumns = mergedArr.shape[1]

	#featureErrorDataList = []
	targetErrData = TargetErrorData(targetName)
	for featureIndex in range(numFeatures):
		sortedArr = getSortedArrayBasedOnColumn(mergedArr,featureIndex)

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
       
#takes, input parameter array, target array and target name
def getRegressionFunctionForEachTarget(inArr, tarArr,tname,degree):
	#print "\n\n--inputArr",inArr,"-----end\n"
        #fit a polynomial regression of degree 2 using Lasso as underlying linear regression model
        #reg = doPolyRegression(inArr, tarArr,tname,3,fitUse="Lasso")   # try for LINPACK and MATRIX MUL
        reg = doPolyRegression(inArr, tarArr,tname,degree,fitUse="Lasso")   # works best
        #reg = doPolyRegression(inArr, tarArr,tname,2,fitUse="LinearRegression") #bad
        #reg = doPolyRegression(inArr, tarArr,tname,2,fitUse="RidgeRegression") #bad
        #print reg
        return reg


def getRegressionFunctionForError(inArr, tarArr,tname,degree):
        #print "\n\n--inputArr",inArr,"-----end\n"
        #fit a polynomial regression of degree 2 using Lasso as underlying linear regression model
        #reg = doPolyRegression(inArr, tarArr,tname,3,fitUse="Lasso")   # try for LINPACK and MATRIX MUL
        #reg = doPolyRegression(inArr, tarArr,tname,degree,fitUse="Lasso")   # works best
        reg = doPolyRegression(inArr, tarArr,tname,degree,fitUse="LinearRegression") #bad
        #reg = doPolyRegression(inArr, tarArr,tname,degree,fitUse="RidgeRegression") #bad
        #print reg
        return reg
 
def curveFitErrorSamplesWithDistance(targetkey,featureName,distanceList,errorList,errorSamples):
	#distanceList = [[1.0],[4.0],[6.0],[9.0]]
	#errorList = [[3.0],[18.0],[38.0],[83.0]]
	distanceArr = np.array(distanceList)
	errArr = np.array(errorList)
	#print getGlobalObject("InArrIndexToColumnIndexMap")
	print "regName = ", targetkey,"  ", featureName
	regName = targetkey + "_" + featureName + "_err"
	#print regName, distanceArr, errArr
	curvFunc = getRegressionFunctionForError(distanceArr,errArr,regName,2)	    
	#curvFunc = getRegressionFunctionForEachTarget(distanceArr,errArr,regName,2)	    
	drawErrorDistPlotWithFittedCurve(errArr,distanceArr,targetkey,featureName,curvFunc,True)	
	drawErrorDistPlot(errArr,distanceArr,targetkey,featureName,True)	
	#testDist = [[11.0]]
	#predVal = curvFunc.predict(testDist)
	#print "Predicted Value (in code test)(123 is correct): ", predVal
	#errDistProfile = errorDistributionProfile(featureName,targetkey)

	#this error profile object is already populated in the map. Just extract that and modify (add errorCurve)
	errDistProfile = getGlobalObject("ErrorDistributionProfileMapForTargetAndFeature")[targetkey][featureName]
	
	errDistProfile.ErrorRegressFunction = curvFunc
	errDistProfile.ErrorSamples = np.array(errorSamples)
	return errDistProfile

def calculateResultantError(errorMap):
	errorObjectForFirstFeature = None
	correlationAdjustedErrorTerms = {}
	#print "use the formula : sqrt[ {1/n(|e1| + |c12e2| + |c13e3| + ..)}^2 + {(1-c12)e2}^2 + {(1-c13)e3}^2 + ...] "
	# n = 1 + c12 + c13
        n = 0.0
	for idx in errorMap.keys():
		(corrCoeff,errorForFeature) = errorMap[idx] 
		if(idx == 1):
			correlationAdjustedErrorTerms[1] = abs(errorForFeature)
			n = n + 1.0
		else:
			#calculate the first term "(e1 + c12e2 + c13e3 + ..)" 
			correlationAdjustedErrorTerms[1] = correlationAdjustedErrorTerms[1] + abs((corrCoeff)*(errorForFeature))
			# now calculate other terms "(1-c12)e2" as "(1-c13)e3"
			correlationAdjustedErrorTerms[idx] = abs((1 - corrCoeff)*(errorForFeature))
			n = n + corrCoeff
	#end for 

	correlationAdjustedErrorTerms[1] = float(correlationAdjustedErrorTerms[1]/n) # normalize	
	#now calculate the RMS of these terms to get resultant error
	sumOfSquares = 0.0
	for key,value in correlationAdjustedErrorTerms.iteritems():
		#print "val = " , value
		sumOfSquares = sumOfSquares + (value)*(value)     #calculate sqrs
	#end for
	rmsError = math.sqrt(sumOfSquares)				

	return rmsError
	

#see "getResultantErrorFromFeatureErrorsForATargetAtADatapoint" func to check how the input map was populated
def getCorrelationBetweenErrorsWRTFirstFeature(productionErrorInfoDict):
	errorCorrMap = {}
	i = 0
	firstErrorSamples = None
	for featureName in productionErrorInfoDict.keys():
		i = i + 1
		(errSamples, errorForFeature) = productionErrorInfoDict[featureName]
		
		if(i==1):
			firstErrorSamples = errSamples
			errorCorrMap[featureName] = (1.0,errorForFeature)
			#print "featreName =", featureName, " corrCoeff = " , "1.0" , " error term = " , errorForFeature
			continue
		else:
			corrCoeff = calculateCorrelationBetweenVectors(firstErrorSamples,errSamples)	
			errorCorrMap[featureName] = (corrCoeff,errorForFeature)
			#print "featreName =", featureName, " corrCoeff = " , corrCoeff , " error term = " , errorForFeature
	#END of for

	return errorCorrMap

	
#pass a featureDataPoint (parameter values found in production) and a target name
#it will return how much prediction error can be there
#returns: rmsError,if all err components were +ve, if all err components were -ve
#if all components were positive, we will consider final result to be only positive and not +/- around predicted targetValue 
def getResultantErrorFromFeatureErrorsForATargetAtADatapoint(targetName,featureDtPt,errProfMap=None):
	#print "\ncalled\n"
	#import traceback
	#traceback.print_stack(file=sys.stdout)
	featureErrMap = None
	if(errProfMap == None):
		featureErrMap = getGlobalObject("ErrorDistributionProfileMapForTargetAndFeature")[targetName]
	else:
		featureErrMap = errProfMap[targetName]

	tempErrorInfoDict = {}
	
	errorTermsAllPositive = True
	errorTermsAllNegative = True

	#print "TEST: featureErrorMap: ", featureErrMap 
	#print "TEST: featureDtPt: ", featureDtPt 

	idx = 0
	#key is feature name, value is value of that feature at the intended location
	for featureName,value in featureDtPt.featureNameValueMap.iteritems():
		idx = idx + 1	
		#get the error profile for this feature	
		errProf = featureErrMap[featureName]
		#print errProf

		#get the distance of that feature from the training location
		meanPoint = errProf.MeanPointOfTrainingSet
		StdDev = errProf.StandardDeviationOfTrainingSet
		distance = getDistanceAlongOneFeature(meanPoint,StdDev,value)
		#get the curve/regression function which fits the variation of error with distance
		errorForFeature = errProf.ErrorRegressFunction.predict(distance)
		#print "targetName", targetName," featureName=",featureName, " mean=",meanPoint," StdDev=",StdDev," distance=",distance," errorForFeature=",errorForFeature
		
		if(errorForFeature > 0.0):
			errorTermsAllNegative = False
		if(errorForFeature < 0.0):
			errorTermsAllPositive = False
		#print "featureName = " , featureName, " sample = ", errProf.ErrorSamples, " err = ", errorForFeature
		#tempErrorInfoDict[featureName] = (errProf.ErrorSamples, errorForFeature) # put raw error samples, and predicted err
		tempErrorInfoDict[idx] = (errProf.ErrorSamples, errorForFeature) # put raw error samples, and predicted err
		#^^ later we will use this info to first calculate correlation between errors and then resultant error
	
	#END of for loop
	#print "TEST: tempErrorInfoDict ", tempErrorInfoDict
	
	#this function calculates the correlation between the error in first feature and all other (n-1) features
	#returns a map. feature name is key, and a tuple (correlation value compared to first, actual error along that feature) 
	featureErrorCorrelaionIndividualErrorMap = getCorrelationBetweenErrorsWRTFirstFeature(tempErrorInfoDict)
	
	#here use the formula : sqrt[ {(e1 + c12e2 + c13e3 + ..)}^2 + {(1-c12)e2}^2 + {(1-c13)e3}^2 + ...]	
	#print "use the formula : sqrt[ {(|e1| + |c12e2| + |c13e3| + ..)}^2 + {(1-c12)e2}^2 + {(1-c13)e3}^2 + ...] "

	rmsError = calculateResultantError(featureErrorCorrelaionIndividualErrorMap)
	
	return (rmsError,errorTermsAllPositive,errorTermsAllNegative)

#X = [[0, 1], [1, 1]]
#getStandardizedEuclideanDistance(X,[[0, 0]])
#A = np.array([[1,11], [0,0], [2,13],[3,12]])
#print "Reference arr: A = ", A 
#getSortedArrayBasedOnColumn(A,1)
#getSortedArrayBasedOnColumn(A,0)
#getSamplesFromBootstrap(A)
#B = [[1.5,2]]
#C = [[30,40]]
#x = getMeanOfObservations(A)
#getStandardizedEuclideanDistance(x,C)
#print "=============== pairwise ================"
#getPairWiseDistance(A,C)
#D = [[300,400]]
#print "D = [[300,400]]"
#getPairWiseDistance(A,D)
#E = [[3000,4000]]
#print "E = [[3000,4000]]"
#getPairWiseDistance(A,E)
#F = [[30000,40000]]
#print "F = [[30000,40000]]"
#getPairWiseDistance(A,F)

if __name__ == "__main__":
	inArr = np.array([[1,11], [0,0], [2,13],[3,12],[4,10],[5,20]])
	tarArr = np.array([[1], [0], [4],[9],[16],[25]])
	targetErrDataStruct = generateTrainingAndTestSetsForDistanceProfilingForEachTarget(inArr,tarArr,"target-1")
	printErrorDataStructureMap(targetErrDataStruct)
