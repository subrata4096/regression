#!/usr/bin/python
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
from scipy.spatial.distance import *
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from micAnalysis import *

import sys
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

#merge two arrays side by side (horizontally)
def getMergedInputAndTargetArray(inArr, targetArr):
	mergedArr = np.concatenate((inArr, targetArr), axis=1)
	print mergedArr
	return mergedArr

#sort an array based on the column index provided
def getSortedArrayBasedOnColumn(inArr,columnIndex):
	sortedArr = inArr[inArr[:,columnIndex].argsort()]
	print "Sorted array, based on column = ", columnIndex
	print sortedArr
	return  sortedArr

#print the error data structure map
def printErrorDataStructureMap(errDSList):
	for errDS in errDSList:
		print "Training obs: \n\t" + str(errDS.TrainingObservations)
		testObs = errDS.TestObservations
		for o in testObs:
			print "Test obs: \n\t" + str(o)
		

#def getSamplesFromBootstrap(inArr, resampleNumber, percentageInTrainSet, useBootstrap):
#	bs = cross_validation.Bootstrap(numSamples, resampleNumber, percentageInTrainSet, random_state=0)
#	print bs
#        testTrainPairMap = {}
#        bootStrapIndex = 0;
#        for train_index, test_index in bs:
#                bootStrapIndex = bootStrapIndex + 1
#                trainList = []
#                testDataPoints = []
#                print("TRAIN:", train_index, "TEST:", test_index)
#                for idx in train_index:
#                        #print idx
#                        trainList.append(inArr[idx])
#                trainArr = np.array(trainList)
#                for idx in test_index:
#                        testDataPoints.append(inArr[idx])
#
#                testTrainPairMap[bootStrapIndex] = (trainArr,testDataPoints)
#
#        #print testTrainPairMap
#        for key in testTrainPairMap.keys():
#                print "Key = ", key, "  value = ", testTrainPairMap[key]
#
#        return testTrainPairMap


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
		print "\n\n== ERROR == ! Please call getSamplesFromBootstrap instead..\n\n"
	else:
		trainingSize = (int)(0.5 + (float)(numSamples * percentageInTrainSet))   #0.5 for rounding up of sample size
		trainSet = [x for x in range(trainingSize)]
		testSet = [x for x in range(trainingSize,numSamples)]
		print trainingSize, trainSet, testSet
		bootStrapIndex = bootStrapIndex + 1
		trainList = []
		testDataPoints = []
  		print("TRAIN:", trainSet, "TEST:", testSet)
		for idx in trainSet:
			#print idx
			trainList.append(inArr[idx])
		trainArr = np.array(trainList)
		for idx in testSet:
			testDataPoints.append(inArr[idx])
		
		testArr = np.array(testDataPoints)
		testTrainPairMap[bootStrapIndex] = (trainArr,testArr)
			
	#print testTrainPairMap
	for key in testTrainPairMap.keys():
		print "Key = ", key, "  value = ", testTrainPairMap[key] 

	return testTrainPairMap

def getStandardizedValues(inArr):

	return

def getMeanOfObservations(inArr):
	#inArr = np.array([[0, 0], [1,11], [2,12],[3,13]])  => format
	#print inArr
        #output:   [ 1.5  9. ]

	centerOfRef = np.mean(inArr,axis=0)
	print centerOfRef
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

def getStandardizedEuclideanDistance(refArr, otherArr):

	#standardized Euclidean distance, meaning that it is the Euclidean distance calculated on standardized data. 

	#For efficiency reasons, the euclidean distance between a pair of row vector x and y is computed as:
	#dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

	distArr = euclidean_distances(refArr, otherArr)
	print distArr
	return distArr	

def getObservationsFromMergedSamples(samples,numOfFeatures,totalNumberOfColumns):
	print "merged sample: " , samples
        #while splitting horizontally, we should give [numOfFeatures, totalNumberOfColumns] argumnet to hsplit
	#which will return 3 arrays:
	# 1. samples[:numOfFeatures] => essntially the data-point part of the merged array 
	# 2. samples[numOfFeatures : totalNumberOfColumns] => essentially the target array
	# 3. samples[totalNumberOfColumns:] => essentially an empty array -> we are not interested
	
	print "Num features: ",numOfFeatures, " total columns: " , totalNumberOfColumns
	splittedList = np.hsplit(samples, np.array([numOfFeatures, totalNumberOfColumns] ))
	paramArr = splittedList[0]  #or inArr or data point array
	targetArr = splittedList[1]
	obs = Observations(paramArr,targetArr)
	return obs
	
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
		trainAndTestSamples = getSamplesFromSortedParams(sortedArr,1,0.66,False)

		feErr = FeatureErrorData()
                feErr.name = featureIndex 
		for key in trainAndTestSamples.keys():
			trainSample,testSamples = trainAndTestSamples[key]
			traingObs = getObservationsFromMergedSamples(trainSample,numFeatures, numTotalColumns)
			feErr.TrainingObservations = traingObs
			for testSample in testSamples:
				testObs = getObservationsFromMergedSamples(testSample,numFeatures, numTotalColumns)
				feErr.TestObservations.append(testObs)
		
		#now append this featureErrorData into the list. This is for a target, 
		#So latter should be attached to the target name
		#featureErrorDataList.append(feErr)  
		targetErrData.FeatureErrorDataMap[feErr.name] = feErr 

	#return featureErrorDataList                 
	return targetErrData               


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
	errDataStruct = generateTrainingAndTestSetsForDistanceProfiling(inArr,tarArr)
	printErrorDataStructureMap(errDataStruct)
