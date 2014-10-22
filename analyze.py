#!/usr/bin/python
from sklearn import linear_model
from sklearn import cross_validation
import numpy as np
from matplotlib import pyplot as plt
import sys
from regressFit import *

#inputColumnNames = ['module:input:0:numRanks','module:input:0:nx']
#inputColumnNames = ['module:input:0:numRanks','module:input:0:nx']
#inputColumnNames = ['module:input:0:ii','module:pub_input::dt','module:pub_input::eKinetic','module:pub_input::ePotential','module:pub_input::iStep','module:pub_input::lat','module:pub_input::momStdDev','module:pub_input::posStdDev']
inputColumnNames = ['module:input:0:ii','module:pub_input::dt','module:pub_input::iStep','module:pub_input::lat']
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

def getRowKey(row):
	s = ""
	for k in row:
		if(s== ""):
			s = str(k)
		else:	
			s = s + ":" + str(k)
	return s


def getAveragePerExperiments(inArr, dataArr):
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
	#print inDict
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
        mean = np.mean(targetArr)
        stddev = np.std(targetArr)
        stdPerMean = mean/stddev
        print "Standard deviation: ", stddev
        print "Standard deviation divided by mean: ", stdPerMean
   
def doFitForTarget(inArr,targetArr, tname):
	#print targetArr

	print "\n******For output:  ", tname
	
    	calculateStatisticOfTarget(targetArr)
	in_train, in_test, tar_train, tar_test = cross_validation.train_test_split(inArr, targetArr, test_size=0.20, random_state=42)

	#print in_train
	#print tar_train
	#print in_test
	#print tar_test
	reg = doLinearRegression(in_train,tar_train)
	print "R2 score: ",reg.score(in_test, tar_test)

	reg = doPolyRegression(in_train, tar_train,tname)
	print "R2 score: ",reg.score(in_test, tar_test)

	#doLinearRegWithCV(in_train, tar_train)
	#doRidgeWithCV(in_train, tar_train)
        #print "Coeff: ",clf.coef_

def scikit_scripts(inArr,measuredArr,outArr):
	i = 0
	for targetArr in measuredArr:
		t = measuredColumnNames[i]
		doFitForTarget(inArr,targetArr,t)
		i = i + 1

	i = 0
	for targetArr in outArr:
		t = outputColumnNames[i]
		doFitForTarget(inArr,targetArr,t)
		i = i + 1
			

#def skll_scripts(inArr,dataArr):
#	import skll
#	learner = skll.Learner('LinearRegression')
	
def getArrayWithUniqueInputs(a):
	unique_a = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
	return unique_a

if __name__ == "__main__":
	dataFile = sys.argv[1]
        
 	inputDataArr = readDataFile(dataFile,'input')
	inputDataArr = np.transpose(inputDataArr)
 	measuredDataArr = readDataFile(dataFile,'measured')
	#print "measured"
	#print measuredDataArr
 	#outputDataArr = readDataFile(dataFile,'output')

	#get an average of values for unique input combinations...        
	uniqueInputArr, averagedMeasuredArr = getAveragePerExperiments(inputDataArr,measuredDataArr)	
	#uniqueInputArr, averagedOutputArr = getAveragePerExperiments(inputDataArr,outputDataArr)
        averagedOutputArr = []
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
