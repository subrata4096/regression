#!/usr/bin/python

from errorDatastructure import *
from errorAnalysis import *
from pickleDump import *
from fields import *
from detectAnomaly import *
#global ErrorDistributionProfileMapForTargetAndFeature
 
def test_errorProfileMapLoad(cPicklepath):
	errProfMap = loadErrorDistributionProfileMap(cPicklepath,True)
	#setGlobalObject("ErrorDistributionProfileMapForTargetAndFeature") = errProfMap
	printErrorDistributionProfileMapForTargetAndFeatureMap(errProfMap)


def getTestPoints():
	testList = []
	testList.append({'module:input:0:dim1':10, 'module:input:0:dim2':10, 'module:input:0:dim3':10})  
	testList.append({'module:input:0:dim1':20, 'module:input:0:dim2':20, 'module:input:0:dim3':20})  
	testList.append({'module:input:0:dim1':50, 'module:input:0:dim2':50, 'module:input:0:dim3':50})  
	testList.append({'module:input:0:dim1':70, 'module:input:0:dim2':70, 'module:input:0:dim3':70})  
	testList.append({'module:input:0:dim1':100, 'module:input:0:dim2':100, 'module:input:0:dim3':100})  
	testList.append({'module:input:0:dim1':120, 'module:input:0:dim2':120, 'module:input:0:dim3':120})  
	testList.append({'module:input:0:dim1':200, 'module:input:0:dim2':200, 'module:input:0:dim3':200}) 
	testList.append({'module:input:0:dim1':300, 'module:input:0:dim2':300, 'module:input:0:dim3':300}) 
	return testList 

def test_resultantError(anomalyDetectObject,targetName):
	print "For target: " + targetName
	testList = getTestPoints()
	for prodDataPointMap in testList:
		print "\n\n Input: " , str(prodDataPointMap)
		fDpt = FeatureDataPoint(prodDataPointMap)
		anomalyDetectObject.getPredictionErrorEstimation(targetName,fDpt)
		valRange = anomalyDetectObject.getValidRangeOfTargetValue(targetName,fDpt)


if __name__ == "__main__" :
	referenceDataFile = sys.argv[1]
	dumpDir = makeDumpDirectory(referenceDataFile)
        setGlobalObject("activeDumpDirectory",dumpDir)

	anoDetectEngine = anomalyDetectionEngine()
        anoDetectEngine.dumpDirectory = dumpDir
        anoDetectEngine.loadPerModuleObjects()

        #anoDetect = anomalyDetection()
        anoDetect = anoDetectEngine.getAnomalyDetectionObject(referenceDataFile)	

	test_resultantError(anoDetect,'module:measure:time:time')
	test_resultantError(anoDetect,'module:measure:PAPI:PAPI_TOT_INS')
	test_resultantError(anoDetect,'module:measure:PAPI:PAPI_L2_DCM')
	#test_errorProfileMapLoad("/home/mitra4/work/regression/errMapDump.cpkl")
	#test_resultantError("/home/mitra4/work/regression/errMapDump.cpkl")
