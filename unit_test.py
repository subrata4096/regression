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

def test_resultantError(anomalyDetectObject):
	targetName = "o3"
	prodDataPointMap = {}
	prodDataPointMap["in1"] = 2
	prodDataPointMap["in2"] = 2
	prodDataPointMap["in3"] = 2
	fDpt = FeatureDataPoint(prodDataPointMap)
	#errProfMap = loadErrorDistributionProfileMap(getGlobalObject("activeDumpDirectory"),True)
	#print errProfMap
	#rmsErr,errPostibeBias,errMegBias = getResultantErrorFromFeatureErrorsForATargetAtADatapoint(targetName,fDpt,errProfMap)
	#print rmsErr, errPostibeBias,errPostibeBias
	anomalyDetectObject.getPredictionErrorEstimation(targetName,fDpt)
	anomalyDetectObject.getValidRangeOfTargetValue(targetName,fDpt)


if __name__ == "__main__" :
	referenceDataFile = sys.argv[1]
	dumpDir = makeDumpDirectory(referenceDataFile)
        setGlobalObject("activeDumpDirectory",dumpDir)
	
	anoDetectEngine = anomalyDetectionEngine()
	anoDetectEngine.dumpDirectory = dumpDir
	anoDetectEngine.loadPerModuleObjects()

	#anoDetect = anomalyDetection()
	anoDetect = anoDetectEngine.getAnomalyDetectionObject(referenceDataFile)
	#anoDetect.errorProfPicklePath = getGlobalObject("activeDumpDirectory")
	#anoDetect.usefulFeaturePicklePath = "/home/mitra4/work/regression/selInput.cpkl"
	#anoDetect.dumpDirectory = getGlobalObject("activeDumpDirectory")

	#anoDetect.loadAnalysisFiles()
	test_resultantError(anoDetect)
	#test_errorProfileMapLoad("/home/mitra4/work/regression/errMapDump.cpkl")
	#test_resultantError("/home/mitra4/work/regression/errMapDump.cpkl")
