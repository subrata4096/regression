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

global inputColumnNames
global measuredColumnNames
global outputColumnNames
global regressionDict


def getRowKey(row):
	s = ""
	for k in row:
		if(s== ""):
			s = str(k)
		else:	
			s = s + ":" + str(k)
	return s


def calculateVariability(inArr, dArr):
	#print "dArr:", dArr
        dataArr = np.transpose(dArr)

        # here we want to get an average of the values that corresponds to same input combinations. May be these are values from multiple experiments..
        inDict =  dict()
        i = -1
        #first create a map where key is the input combination and values are list of output arrays
        for row in inArr:
                i = i + 1
                rowKey = getRowKey(row)
                if rowKey in inDict.keys():
                        inDict[rowKey].append(dataArr[i])
                else:
                        inDict[rowKey] = []
                        inDict[rowKey].append(dataArr[i])
        #print inDict.keys()
        #dataDict = map(lambda t: list(t), (inDict[key]  for key in inDict.keys()))
        #dataDict = [a.tolist() for a in (inDict[key]  for key in inDict.keys())]
        h1 = []
        for key in inDict.keys() :
                h2 = []
                for s in inDict[key]:
                        h2.append(list(s))
                h1.append(h2)
        #print " printing h1:"
        #print "h1", h1
        #dDict = [l.tolist() for l in dataDict]
        # get the average...
        #A = [np.average(x, axis=0) for x in h1]
        meanA = [np.mean(x, axis=0) for x in h1]
        stdDevA = [np.std(x, axis=0) for x in h1]
        variance = [np.var(x, axis=0) for x in h1]
        stdOverMean = [np.std(x, axis=0)/np.mean(x, axis=0) for x in h1]
        #stdOverMean = [np.std(x, axis=0) for x in h1]
        #print "mean:", meanA 
        #print "stddev:", stdDevA 
        #print "var:", variance
        #print "stdOverMean:", stdOverMean
	
	outStat = {}
	numVals = {}
	for val in stdOverMean:
		print val
		for i in range(len(dArr)):
			if i not in outStat.keys():
				outStat[i] = 0
				numVals[i] = 0
			if(False == np.isfinite(val[i])):
				continue   #avoid infinity in stdOverMean which will happen if mean = 0 
			outStat[i] = outStat[i] + val[i]
			numVals[i] = numVals[i] + 1
	for key in outStat.keys():
		if(0 == numVals[key]):
			outStat[key] = np.inf
		else:
			outStat[key] = outStat[key]/(float(numVals[key]))
		
	#print "outStat: ", outStat
	return outStat

        # convert a list of 1D arrays to a 2D array
        #A = map(lambda t: list(t), A)
        #newDataArr = np.array(A)
        #print newDataArr

        #inList = [key.split(':') for key in inDict.keys()]
        #newInputArr = np.array(inList,dtype=float)


def getAveragePerExperiments(inArr, dataArr):

	return inArr, dataArr # do not perform any average. let the regressor handle it

	dataArr = np.transpose(dataArr)
	#print inArr
	#print dataArr

	# here we want to get an average of the values that corresponds to same input combinations. May be these are values from multiple experiments..
	inDict =  dict()
	i = -1
	#first create a map where key is the input combination and values are list of output arrays
	for row in inArr:
		i = i + 1
		rowKey = getRowKey(row)
		if rowKey in inDict.keys():
			inDict[rowKey].append(dataArr[i])
		else:
			inDict[rowKey] = []
			inDict[rowKey].append(dataArr[i])
	#print inDict.keys()
	#dataDict = map(lambda t: list(t), (inDict[key]  for key in inDict.keys()))
	#dataDict = [a.tolist() for a in (inDict[key]  for key in inDict.keys())]
	h1 = []
	for key in inDict.keys() :
		h2 = []
		for s in inDict[key]:
			h2.append(list(s))
		h1.append(h2)
	#print " printing h1:"
	print h1
	#dDict = [l.tolist() for l in dataDict]
	# get the average...
	A = [np.average(x, axis=0) for x in h1]
	#meanA = [np.mean(x, axis=0) for x in h1]
	#stdDevA = [np.std(x, axis=0) for x in h1]
	#print "mean:", meanA, " stddev:", stdDevA 
	# convert a list of 1D arrays to a 2D array
	A = map(lambda t: list(t), A)
	newDataArr = np.array(A)
	#print newDataArr
	
	inList = [key.split(':') for key in inDict.keys()]
	newInputArr = np.array(inList,dtype=float)

	newDataArr = np.transpose(newDataArr)

	return newInputArr, newDataArr			
		

def readDataFile(fName,field_type):
	indexes = []
	if(field_type == 'input'):
		indexes = getColumnIndexes(fName,inputColumnNames)
   	elif(field_type == 'measured'):
		indexes = getColumnIndexes(fName,measuredColumnNames)
	elif(field_type == 'output'):
		indexes = getColumnIndexes(fName,outputColumnNames)
	
	print indexes	
	realdata = np.loadtxt(fName, dtype=float,delimiter='\t', usecols=indexes, converters=None,skiprows=1)
	data = np.transpose( realdata)
	#print data
	return data
	
def getColumnIndexes(fName, columnNames):
	if(len(columnNames) == 0):
		return []
	#data = np.genfromtxt(fName, dtype=None, delimiter='\t', names=True)
	d = np.genfromtxt(fName, dtype=str,delimiter='\t')
		
	names = d[0]
	#print names
	indexes = []
	i = 0
	for name in names:
		if(name in columnNames):
			indexes.append(i)
		i = i + 1
	#print indexes
	return indexes

def calculateStatisticOfTarget(targetArr):
	scaled_target = preprocessing.scale(targetArr)
        mean = np.mean(scaled_target)
        stddev = np.std(scaled_target)
        #stdPerMean = mean/stddev
        print "Standard deviation: ", stddev
        print "Mean: ", mean
        #print "Standard deviation divided by mean: ", stdPerMean
   
def doFitForTarget(inArr,targetArr, tname):
	#print targetArr

	print "\n******For output:  ", tname

	#just use one input for MIC testing purpose
        #inArr = inArr[:,1]
	#inArr = inArr[:,None]
	#print inArr	

    	#calculateStatisticOfTarget(targetArr)

	#Do MIC analysis based on Science 2011 paper
        selected_inArr = doMICAnalysisOfInputVariables(inArr, targetArr)
	#print "sel list" , selected_inArr


	if(len(selected_inArr) == 0):
		print "No input captures the target:",tname
		return None

	in_train, in_test, tar_train, tar_test = cross_validation.train_test_split(selected_inArr, targetArr, test_size=0.30, random_state=42)

	#print in_train
	#print tar_train
	#print in_test
	#print tar_test
	#reg = doLinearRegression(in_train,tar_train)
	#print "R2 score: ",reg.score(in_test, tar_test)

	scaled_in_train = preprocessing.scale(in_train)
	scaled_tar_train = preprocessing.scale(tar_train)
	
	scaled_in_test = preprocessing.scale(in_test)
	scaled_tar_test = preprocessing.scale(tar_test)

	#reg = doPolyRegression(in_train, tar_train,tname,2,fitUse="LinearRegression")
	#print "R2 score: ",reg.score(in_test, tar_test)
	#reg = doPolyRegression(in_train, tar_train,tname,2,fitUse="RidgeRegression")
	#print "R2 score: ",reg.score(in_test, tar_test)
	reg = doPolyRegression(in_train, tar_train,tname,2,fitUse="Lasso")
	print "R2 score: ",reg.score(in_test, tar_test)
	#reg = doPolyRegression(in_train, tar_train,tname,2,fitUse="ElasticNet")
	#print "R2 score: ",reg.score(in_test, tar_test)

	#reg = doRidgeWithCV(in_train, tar_train)
	#print "R2 score: ",reg.score(in_test, tar_test)

	#reg = doLinearRegWithCV(in_train, tar_train)
        #print "Coeff: ",clf.coef_
	#doPlot(inArr,targetArr,4,tname,reg)
	#do3dPlot(inArr,targetArr,4,9,tname,reg)

	return reg


def scikit_scripts(inArr,measuredArr,outArr):
	i = 0
	for targetArr in measuredArr:
		t = measuredColumnNames[i]
		reg = doFitForTarget(inArr,targetArr,t)
		regressionDict[t] = reg
		i = i + 1

	i = 0
	for targetArr in outArr:
		t = outputColumnNames[i]
		reg = doFitForTarget(inArr,targetArr,t)
		regressionDict[t] = reg
		i = i + 1
			

#def skll_scripts(inArr,dataArr):
#	import skll
#	learner = skll.Learner('LinearRegression')
	
def getArrayWithUniqueInputs(a):
	unique_a = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
	return unique_a

if __name__ == "__main__":
	dataFile = sys.argv[1]
	productionDataFile = ""
	#productionDataFile = sys.argv[2]
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

	measureVari = calculateVariability(inputDataArr,measuredDataArr)
	outVari = calculateVariability(inputDataArr,outputDataArr)
	
	#get an average of values for unique input combinations...        
        averagedOutputArr = []
	averagedMeasuredArr = []
	uniqueInputArr, averagedMeasuredArr = getAveragePerExperiments(inputDataArr,measuredDataArr)	
	uniqueInputArr1, averagedOutputArr = getAveragePerExperiments(inputDataArr,outputDataArr)
	#print uniqueInputArr
	#print "\n------"
	#print averagedMeasuredArr
	#get the an array with only unique input combinations (array with unique rows)
	#uniqueInputArr = getArrayWithUniqueInputs(inputDataArr)
	#print "\n\n ----- "
	#print uniqueInputArr
	#print "\n\n ----- "
        #print averagedMeasuredArr
	#print "\n\n ----- "
	scikit_scripts(uniqueInputArr,averagedMeasuredArr,averagedOutputArr)

	if(productionDataFile != ""):
		prodInputArr = readDataFile(productionDataFile,'input')
        	prodInputArr = np.transpose(prodInputArr)
        	prodMeasureArr = readDataFile(productionDataFile,'measured')
        	prodOutputArr = readDataFile(productionDataFile,'output')
        	prodOutputArr = []
		prodInputArr, prodMeasureArr = getAveragePerExperiments(prodInputArr, prodMeasureArr)
		anomaly_detection(prodInputArr, prodMeasureArr, prodOutputArr)
