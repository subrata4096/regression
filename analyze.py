#!/usr/bin/python
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt
import sys

#inputColumnNames = ['module:input:0:numRanks','module:input:0:nx']
inputColumnNames = ['in1', 'in2', 'in3']
#measuredColumnNames = ['module:measure:RAPL:Elapsed','module:measure:RAPL:EDP_S0']
measuredColumnNames = ['m1','m2']
#outputColumnNames = ['module:output:0:TotalAbsDiff','module:output:1:numCycles']
outputColumnNames = ['o1','o2']

def getRowKey(row):
	s = ""
	for k in row:	
		s = s + ":" + str(k)
	return s

def getAveragePerExperiments(inArr, dataArr):
	dataArr = np.transpose(dataArr)
	#print inArr
	#print dataArr
	inDict =  dict()
	i = -1
	for row in inArr:
		i = i + 1
		rowKey = getRowKey(row)
		if rowKey in inDict.keys():
			inDict[rowKey].append(dataArr[i])
		else:
			inDict[rowKey] = []
			inDict[rowKey].append(dataArr[i])
	inDict = map(lambda t: list(t), (inDict[key]  for key in inDict.keys()))
	print inDict
	#np.average(x) for x in inDict
	#arrs = [arr for arr in (inDict[key]  for key in inDict.keys())]
	#A = map(lambda t: list(t), arrs)
	#print A			
		

def readDataFile(fName,field_type):
	indexes = []
	if(field_type == 'input'):
		indexes = getColumnIndexes(fName,inputColumnNames)
   	elif(field_type == 'measured'):
		indexes = getColumnIndexes(fName,measuredColumnNames)
	elif(field_type == 'output'):
		indexes = getColumnIndexes(fName,outputColumnNames)
		
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
   
def doFitForTarget(inArr,targetArr, tname):
	#print targetArr

	#first add a column of 1s so that it can be used as constant for linear regression
	idx = len(inArr[0])
	inArrWithConst = np.insert(inArr,idx, 1, axis=1)
	#print inArrWithConst


	#print inArr.shape, targetArr.shape
	#print "For output:  ", targetArr, " ------------------------------"
	print "For output:  ", tname
	#clf = linear_model.Lasso(normalize=True)
	#clf = linear_model.LinearRegression( normalize=True)
	clf = linear_model.LinearRegression(fit_intercept=False) #fit_intercept=False is very important. prevents the LinearRegression object from working with x - x.mean(axis=0)
        clf.fit (inArrWithConst,targetArr)
        print clf.score(inArrWithConst,targetArr)
	print clf.coef_
	#print np.dot(inArrWithConst, clf.coef_) # reconstruct the output
	# show it on the plot
	plt.plot(inArr, targetArr, label='true data')
	#plt.plot(testY, predSvr, 'co', label='SVR')
	#plt.plot(testY, predLog, 'mo', label='LogReg')
	plt.legend()
	#plt.show()

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
 	outputDataArr = readDataFile(dataFile,'output')
        
	getAveragePerExperiments(inputDataArr,outputDataArr)	
	p = getArrayWithUniqueInputs(inputDataArr)
	print p
	scikit_scripts(inputDataArr,measuredDataArr,outputDataArr)
