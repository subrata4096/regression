#!/usr/bin/python

from errorDatastructure import *
from errorAnalysis import *
from pickleDump import *
from fields import *
global ErrorDistributionProfileMapForTargetAndFeature
 
def test_errorProfileMapLoad(cPicklepath):
	errProfMap = loadErrorDistributionProfileMap(cPicklepath,True)
	#ErrorDistributionProfileMapForTargetAndFeature = errProfMap
	printErrorDistributionProfileMapForTargetAndFeatureMap(errProfMap)

def test_resultantError(cPicklepath):
	targetName = "o3"
	prodDataPointMap = {}
	prodDataPointMap["in1"] = 20
	prodDataPointMap["in2"] = 20
	prodDataPointMap["in3"] = 20
	fDpt = FeatureDataPoint(prodDataPointMap)
	errProfMap = loadErrorDistributionProfileMap(cPicklepath,True)
	rmsErr,errPostibeBias,errMegBias = getResultantErrorFromFeatureErrorsForATargetAtADatapoint(targetName,fDpt,errProfMap)
	print rmsErr, errPostibeBias,errPostibeBias


if __name__ == "__main__" :
	#test_errorProfileMapLoad("/home/mitra4/work/regression/errMapDump.cpkl")
	test_resultantError("/home/mitra4/work/regression/errMapDump.cpkl")
