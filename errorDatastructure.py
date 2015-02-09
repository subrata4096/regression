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
		s = str("PARAM Arr: " + np.array_str(self.ParamArr) + " TARGET Arr: " + np.array_str(self.TargetArr))
		return s

class FeatureErrorData:
	def __init__(self):
		self.name = ''
		self.TrainingObservations = None
		self.TestObservations = []
	#def __str__(self):
	#	s = "Feature id: " + self.name + "\n" + \
	#		"\t" + "Training obs: " + repr(self.TrainingObservations.str()) + "\n" 
	#		#+ "\t" + "Test obs: " + "\n"  
	#		#+ "\t\t" + self.TestObservations.str() + "\n")
	#	return s
  

class TargetErrorData:
	def __init__(self):
		self.name = ''
		# featureName vs FeatureErrorData map
		self.FeatureErrorDataMap = {}
	#def __str__(self):
	#	s = "TargetErrorData" + str(self.FeatureErrorDataMap)
	#	return s

#targetName vs TargetErrorData map
TargetErrorDataMap = {}
	
if __name__ == "__main__":
	o = FeatureErrorData()
