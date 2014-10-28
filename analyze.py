#!/usr/bin/python
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
import sys
from regressFit import *
from micAnalysis import *

inputColumnNames = []
measuredColumnNames = []
outputColumnNames = []

#inputColumnNames = ['module:input:0:numRanks','module:input:0:nx']
#inputColumnNames = ['module:input:0:numRanks','module:input:0:nx']
#inputColumnNames = ['module:input:0:balance','module:input:0:cost','module:input:0:dataStruct','module:input:0:dtfixed','module:input:0:dtmax','module:input:0:host','module:input:0:its','module:input:0:numNodes','module:input:0:numRanks', 'module:input:0:numReg','module:input:0:numZones', 'module:input:0:nx','module:input:0:powercap','module:input:0:rank','module:input:0:real_prec','module:input:0:system','module:input:1:FOM','module:input:1:MaxAbsDiff','module:input:1:MaxRelDiff','module:input:1:TotalAbsDiff','module:input:1:iter','module:input:1:numElem','module:input:1:numNode','module:input:1:phase','module:input:1:u_cut']
#inputColumnNames = ['module:pub_input::iStep','module:pub_input::lat']
inputColumnNames = ['module:input:0:ii','module:pub_input::dt','module:pub_input::eKinetic','module:pub_input::ePotential','module:pub_input::iStep','module:pub_input::lat','module:pub_input::momStdDev','module:pub_input::posStdDev']
#inputColumnNames = ['module:input:0:ii','module:pub_input::dt','module:pub_input::iStep','module:pub_input::lat']
#inputColumnNames = ['module:pub_input::dt','module:pub_input::lat','module:input:0:iStep']
#inputColumnNames = ['module:pub_input::dt','module:pub_input::lat']
#inputColumnNames = ['module:input:0:dt','module:input:0:lat']
#inputColumnNames = ['in1', 'in2', 'in3']
#measuredColumnNames = ['module:measure:PAPI:PAPI_L2_TC_MR','module:measure:PAPI:PAPI_TOT_INS','module:measure:RAPL:Elapsed']
measuredColumnNames = ['module:measure:PAPI:PAPI_TOT_INS','module:measure:time:time']
#measuredColumnNames = ['module:measure:RAPL:Elapsed','module:measure:RAPL:EDP_S0']
#measuredColumnNames = ['m1','m2']
#outputColumnNames = ['module:output:0:TotalAbsDiff','module:output:1:numCycles']
#outputColumnNames = ['module:output:0:TotalAbsDiff','module:output:1:numCycles']
#outputColumnNames = ['o1','o2']

regressionDict = {}

def getRowKey(row):
	s = ""
	for k in row:
		if(s== ""):
			s = str(k)
		else:	
			s = s + ":" + str(k)
	return s


def getAveragePerExperiments(inArr, dataArr):
	#return inArr, dataArr
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
	#print h1
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

    	calculateStatisticOfTarget(targetArr)
	in_train, in_test, tar_train, tar_test = cross_validation.train_test_split(inArr, targetArr, test_size=0.10, random_state=42)

	#Do MIC analysis based on Science 2011 paper
	doMICAnalysisOfInputVariables(inArr, targetArr)
	
	#print in_train
	#print tar_train
	#print in_test
	#print tar_test
	#reg = doLinearRegression(in_train,tar_train)
	#print "R2 score: ",reg.score(in_test, tar_test)

	#scaled_in_train = preprocessing.scale(in_train)
	#scaled_tar_train = preprocessing.scale(tar_train)
	
	#scaled_in_test = preprocessing.scale(in_test)
	#scaled_tar_test = preprocessing.scale(tar_test)

	#reg = doPolyRegression(in_train, tar_train,tname,fitUse="LinearRegression")
	#print "R2 score: ",reg.score(in_test, tar_test)
	reg = doPolyRegression(in_train, tar_train,tname,fitUse="RidgeRegression")
	print "R2 score: ",reg.score(in_test, tar_test)
	#reg3 = doPolyRegression(in_train, tar_train,tname,fitUse="Lasso")
	#print "R2 score: ",reg3.score(in_test, tar_test)
	#reg4 = doPolyRegression(in_train, tar_train,tname,fitUse="ElasticNet")
	#print "R2 score: ",reg4.score(in_test, tar_test)

	#reg = doRidgeWithCV(in_train, tar_train)
	#print "R2 score: ",reg.score(in_test, tar_test)

	#reg = doLinearRegWithCV(in_train, tar_train)
        #print "Coeff: ",clf.coef_
	return reg

def check_anomaly(production_inArr, targetArr,tname):
	reg = regressionDict[tname]
	#print "Input arr: ", production_inArr
	predicted = reg.predict(production_inArr)
	#print tname, " R2 score: ",reg.score(production_inArr, targetArr)
	#print tname, " prediction: ",reg.predict(production_inArr)
	#print tname, " Actual: ", targetArr
	error = (targetArr[0] - predicted[0])*1.0/float(targetArr[0])
	print "Percentage error: " , error

def anomaly_detection(inArr, measuredArr, outArr):
	i = 0
        for targetArr in measuredArr:
                t = measuredColumnNames[i]
                check_anomaly(inArr,targetArr,t)
                i = i + 1

        i = 0
        for targetArr in outArr:
                t = outputColumnNames[i]
                check_anomaly(inArr,targetArr,t)
                i = i + 1

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
	productionDataFile = sys.argv[2]
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

	#get an average of values for unique input combinations...        
        averagedOutputArr = []
	uniqueInputArr, averagedMeasuredArr = getAveragePerExperiments(inputDataArr,measuredDataArr)	
	#junkInputArr, averagedOutputArr = getAveragePerExperiments(inputDataArr,outputDataArr)
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


	prodInputArr = readDataFile(productionDataFile,'input')
        prodInputArr = np.transpose(prodInputArr)
        prodMeasureArr = readDataFile(productionDataFile,'measured')
        prodOutputArr = readDataFile(productionDataFile,'output')
        prodOutputArr = []
	prodInputArr, prodMeasureArr = getAveragePerExperiments(prodInputArr, prodMeasureArr)
	anomaly_detection(prodInputArr, prodMeasureArr, prodOutputArr)
