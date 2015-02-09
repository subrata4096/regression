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

global inputColumnNames
global measuredColumnNames
global outputColumnNames
global regressionDict

class Observations:
	def __init__(self,inArr,tarArr):
		self.ParamArr = inArr
		self.TargetArr = tarArr
	def __str__(self):
		s = str("\n\tPARAM Arr: " + np.array_str(self.ParamArr) + "\n\tTARGET Arr: " + np.array_str(self.TargetArr))
		return s

class FeatureErrorData:
	def __init__(self):
		self.name = ''
		self.TrainingObservations = None
		self.TestObservations = []
	def __str__(self):
		s = "\n\t\t-----------------  Feature id: " + str(self.name) + " --------\n"
		s = s +	"\t\t\t" + "Training obs: " + str(self.TrainingObservations) + "\n" 
		s = s + "\t\t" + "Test obs: " 
                for testObs in self.TestObservations:
                        s = s + "\n\t\t\t" + str(testObs) 
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

#targetName vs TargetErrorData map
TargetErrorDataMap = {}


def printFullErrorDataStructure():
	print "\n\n******* Target error data map ********"
	for targetkey in TargetErrorDataMap.keys():
		tarErrData = TargetErrorDataMap[targetkey]
		print "For target = ", str(tarErrData)
        #for errDS in errDSList:
        #        print "Training obs: \n\t" + str(errDS.TrainingObservations)
        #        testObs = errDS.TestObservations
        #        for o in testObs:
        #                print "Test obs: \n\t" + str(o)
	
if __name__ == "__main__":
	o = FeatureErrorData()
