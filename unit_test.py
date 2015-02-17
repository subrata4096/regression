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
	targetName = "m1"
	prodDataPointMap = {}
	prodDataPointMap["in1"] = 20
	prodDataPointMap["in2"] = 20
	prodDataPointMap["in3"] = 20
	fDpt = FeatureDataPoint(prodDataPointMap)
	#errProfMap = loadErrorDistributionProfileMap(cPicklepath,True)
	#print errProfMap
	#rmsErr,errPostibeBias,errMegBias = getResultantErrorFromFeatureErrorsForATargetAtADatapoint(targetName,fDpt,errProfMap)
	#print rmsErr, errPostibeBias,errPostibeBias
	anomalyDetectObject.getPredictionErrorEstimation(targetName,fDpt)


if __name__ == "__main__" :
	anoDetect = anomalyDetection()
	anoDetect.errorProfPicklePath = "/home/mitra4/work/regression/errMapDump.cpkl"
	anoDetect.usefulFeaturePicklePath = "/home/mitra4/work/regression/selInput.cpkl"
	anoDetect.loadAnalysisFiles()
	test_resultantError(anoDetect)
	#test_errorProfileMapLoad("/home/mitra4/work/regression/errMapDump.cpkl")
	#test_resultantError("/home/mitra4/work/regression/errMapDump.cpkl")
