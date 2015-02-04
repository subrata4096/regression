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

def getSamplesFromBootstrap(inArr):
	numSamples = inArr.shape[0]
	bs = cross_validation.Bootstrap(numSamples, 10, 0.5, random_state=0)
	testList = []
	for test_index, other_index in bs:
  		print("TRAIN:", test_index, "TEST:", other_index)
		for idx in test_index:
			print idx
			testList.append(inArr[idx])
	testArr = np.array(testList)
	print testArr

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


#X = [[0, 1], [1, 1]]
#getStandardizedEuclideanDistance(X,[[0, 0]])
A = np.array([[0, 0], [1,11], [2,12],[3,13]])
print "Reference arr: A = ", A 
getSamplesFromBootstrap(A)
B = [[1.5,2]]
C = [[30,40]]
x = getMeanOfObservations(A)
getStandardizedEuclideanDistance(x,C)
print "=============== pairwise ================"
getPairWiseDistance(A,C)
D = [[300,400]]
print "D = [[300,400]]"
getPairWiseDistance(A,D)
E = [[3000,4000]]
print "E = [[3000,4000]]"
getPairWiseDistance(A,E)
F = [[30000,40000]]
print "F = [[30000,40000]]"
getPairWiseDistance(A,F)
