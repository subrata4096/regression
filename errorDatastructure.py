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
from micAnalysis import *

import sys
from regressFit import *
from micAnalysis import *
from drawPlot import *
from detectAnomaly import *
from fields import *
from analyze import *
from pickleDump import *

#global inputColumnNames
#global measuredColumnNames
#global outputColumnNames
#global regressionDict

#this is a global container for error datastructure
#targetName vs TargetErrorData map
#global TargetErrorDataMap
#global ErrorDistributionProfileMapForTargetAndFeature
#This is a map where initial samples for test and train are kept.
#Later it is populated with prediction functions and prediction errors.
#The top level key is "target-name". The content DataStructure "TargetErrorData"
#TargetErrorDataMap ={}
#This map keeps the calculated error profile for each target for each profile along with curve-fitted error function
#top level key is target name. 2nd level key is feature name. Then the content is "errorDistributionProfile"
#ErrorDistributionProfileMapForTargetAndFeature = {}

class FeatureDataPoint:
	def __init__(self,nameValueMap):
		self.featureNameValueMap = nameValueMap
	def __str__(self):
		s = str(self.featureNameValueMap)
		return s

def getSelectedFeaturePtFromProductionDataPoint(prodFeaturePt,selectedFeatureMap):
	#print "TEST: prodFeaturePt ", prodFeaturePt
	#print "TEST: selectedFeatureMap ",selectedFeatureMap 
        selectedDataPointMap = {}
        #selectedDataPointMap["in1"] = 2
        selectedFeatures = selectedFeatureMap.keys()
        productionDataPtMap = prodFeaturePt.featureNameValueMap
        #print FeatureDataPoint
        for feature,value in productionDataPtMap.iteritems():
                if feature in selectedFeatures:
                        selectedDataPointMap[feature] = value
	#print "TEST : selectedDataPointMap ", selectedDataPointMap
        dataPtWithSelectedFeature = FeatureDataPoint(selectedDataPointMap)
        return dataPtWithSelectedFeature

def getSelectedInputArrFromSelectedDataPoint(selectedFeaturePt,selectedFeatureSortedByInIndex):
	inArr = []
	# remember, selectedFeatureMap should have a format: <featureName> <=> <index in input feature Arr>
	#AND it is supposed to be sorted based on index in input feature Arr in increasing order
	for selectedFeature in selectedFeatureSortedByInIndex:
		inArr.append(selectedFeaturePt.featureNameValueMap[selectedFeature[0]])
	#end for
	return inArr


class Observations:
	#while creating the observations, specify observationType, TRAIN or TEST. 
	# depending on this type, observation data is populated differently
	# also, PredictedArr and PredictionErrArr is only calculated for observationType=TEST
	def __init__(self,inArr,tarArr,observationType):
                self.observeType = observationType   #TRAIN or TEST. Certain operations are valid on TRAIN and certain are valid on TEST
		self.ParamArr = inArr
		self.TargetArr = tarArr
		self.PredictedArr = None
		self.PredictionErrArr = None
		self.DistanceToTargetArr = None
	def __str__(self):
		s = str("\n\tObservationType:" + self.observeType + "\tPARAM Arr: " + np.array_str(self.ParamArr) + "\tTARGET Arr: " + np.array_str(self.TargetArr))
		if(self.PredictionErrArr != None):
			s = s + "\tPREDICTED Arr: " + np.array_str(self.PredictedArr)
			s = s + "\tPREDICTION ERROR: " + str(self.PredictionErrArr)
			s = s + "\tDISTANCE: " + str(self.DistanceToTargetArr)
		return s

class FeatureErrorData:
	def __init__(self):
		self.name = ''
		self.TrainingObservations = None
		self.TestObservations = []
		self.RegressionFunction = None
	def __str__(self):
		featureName = getInputParameterNameFromFeatureIndex(self.name)
		#s = "\n\t\t-----------------  Feature id: " + str(self.name) + " --------\n"
		s = "\n\t\t-----------------  Feature id: " + featureName + " --------\n"
		s = s + "- - - - - - - - - - - - - - - - - - \n"
		s = s +	"\t\t" + "TRAIN OBSERVATIONS: " + str(self.TrainingObservations) + "\n" 
		s = s + "\t\t" + "TEST OBSERVATIONS: " 
                for testObs in self.TestObservations:
                        s = s + "\n\t\t\t" + str(testObs)
		#s = s + "\n\t\t RegressionFunc:  " + str(self.RegressionFunction) 
		s = s + "\n- - - - - - - - - - - - - - - - - - "
		return s
  
#data structure per target for train and test samples
#the internal FeatureErrorDataMap contains list of training samples and test samples for each (sorted)feature 
class TargetErrorData:
	def __init__(self):
		self.name = ''
		# featureName vs FeatureErrorData map
		self.FeatureErrorDataMap = {}
	def __init__(self,targetName):
		self.name = targetName
		self.FeatureErrorDataMap = {}
	def __str__(self):
		s = "\tTargetErrorData = " + self.name
		for fKeys in self.FeatureErrorDataMap.keys():
			fErrData = self.FeatureErrorDataMap[fKeys]
			s = s + str(fErrData)
		
		s = s + "\n"	 
		return s


def printFullErrorDataStructure():
	print "\n\n******* Target error data map ********"
	tgtErrMap = getGlobalObject("TargetErrorDataMap")
	for targetkey in tgtErrMap.keys():
		tarErrData = tgtErrMap[targetkey]
		print "\n*****************For target = ", str(tarErrData)
        #for errDS in errDSList:
        #        print "Training obs: \n\t" + str(errDS.TrainingObservations)
        #        testObs = errDS.TestObservations
        #        for o in testObs:
        #                print "Test obs: \n\t" + str(o)

class errorDistributionProfile:
	def __init__(self,feature_name,target_name):
		self.FeatureName = feature_name
		self.TargetName = target_name
		self.ErrorRegressFunction = None
		self.MeanPointOfTrainingSet = None
		self.StandardDeviationOfTrainingSet = None
		self.ErrorSamples = []
	def __str__(self):
		s = "\nError profile:  TargetName = " + self.TargetName + " FeatureName = " + self.FeatureName 
		s = s + "\n\tMean point of training = " + str(self.MeanPointOfTrainingSet) + " StdDev of training set = " + str(self.StandardDeviationOfTrainingSet)
		#s = s + "\n\tError Curve Fitted Fuction: " + str(self.ErrorRegressFunction.coeff_)
		s = s + "\n\tError Curve Fitted Fuction: " + str(self.ErrorRegressFunction.named_steps['linear'].coef_)
		s = s + "\n\tError Samples: " + np.array_str(self.ErrorSamples) + "\n"
		#doHistogramPlot(self.ErrorSamples,self.TargetName,self.FeatureName,doSave=True)
		return s
		 
def printErrorDistributionProfileMapForTargetAndFeatureMap(errProfMap=None):
	workingErrProfileMap = errProfMap
	if(errProfMap == None):
		workingErrProfileMap = getGlobalObject("ErrorDistributionProfileMapForTargetAndFeature")
	else:
		workingErrProfileMap = errProfMap
	#print "just printing the map\n", workingErrProfileMap
	for targetKey in workingErrProfileMap.keys():	
		featureErrMap = workingErrProfileMap[targetKey]
		for featureName in featureErrMap.keys():
			errProf = featureErrMap[featureName]
			print "\nxxxxxxxxxxxxxxxx     ERROR PROFILE   xxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
			print str(errProf)
if __name__ == "__main__":
	o = FeatureErrorData()
