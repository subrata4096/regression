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


def testProductionFile(anomalyDetectObject,prodDataFile,msrMap,outMap,fDpt,lineNum,isBasic):
	for targetName in msrMap.keys():
		targetValue = msrMap[targetName]
		testForEachTarget(anomalyDetectObject,targetName,targetValue,fDpt,lineNum,isBasic)

	for targetName in outMap.keys():
                targetValue = outMap[targetName]
                testForEachTarget(anomalyDetectObject,targetName,targetValue,fDpt,lineNum,isBasic)

def testForEachTarget(anomalyDetectObject,targetName,targetValue,fDpt,lineNum,isBasic):
	if(isBasic == False):
		if(targetName not in anomalyDetectObject.goodTargetMap.keys()):
			return	
	#anomalyDetectObject.getPredictionErrorEstimation(targetName,fDpt)
	if(isBasic == False):
		predictedValueErrorAdjusted = anomalyDetectObject.getValidRangeOfTargetValue(targetName,fDpt)
		if((targetValue < predictedValueErrorAdjusted[0]) or (targetValue > predictedValueErrorAdjusted[1])):
			print "\nADVANCED:Error for target=", targetName, "value=", targetValue, " at lineNum=", lineNum," predicted range = ", predictedValueErrorAdjusted, "\n"
			#exit(0)
		else:
			print "\nADVANCED:Allright for target=", targetName, "value=", targetValue, " at lineNum=", lineNum
		#	print "predicted range = ", predictedValueErrorAdjusted, "\n"
	else:
		predictedValueErrorAdjustedBasic = anomalyDetectObject.getValidRangeOfTargetValueBasic(targetName,fDpt)
		if((targetValue < predictedValueErrorAdjustedBasic[0]) or (targetValue > predictedValueErrorAdjustedBasic[1])):
			print "\nBASIC:Error for target=", targetName, "value=", targetValue, " at lineNum=", lineNum," predicted range = ", predictedValueErrorAdjustedBasic, "\n"
	
		else:
			print "\nBASIC:Allright for target=", targetName, "value=", targetValue, " at lineNum=", lineNum
#def test_resultantError(anomalyDetectObject):
#	targetName = "o3"
#	prodDataPointMap = {}
#	prodDataPointMap["in1"] = 2
#	prodDataPointMap["in2"] = 2
#	prodDataPointMap["in3"] = 2
#	fDpt = FeatureDataPoint(prodDataPointMap)
#	#errProfMap = loadErrorDistributionProfileMap(getGlobalObject("activeDumpDirectory"),True)
#	#print errProfMap
#	#rmsErr,errPostibeBias,errMegBias = getResultantErrorFromFeatureErrorsForATargetAtADatapoint(targetName,fDpt,errProfMap)
#	#print rmsErr, errPostibeBias,errPostibeBias
#	anomalyDetectObject.getPredictionErrorEstimation(targetName,fDpt)
#	anomalyDetectObject.getValidRangeOfTargetValue(targetName,fDpt)



def createFeatureDataPointFromProductionFile(anomalyDetectObject,productionFile,isBasic):
	#print "\n\n ++++++ starting \n"
        inColNames,msrColNames,outColNames = parseFields(productionFile)

        #inputDataArr = readDataFile(dataFile,'input')
        #measuredDataArr = readDataFile(dataFile,'measured')
        #outputDataArr = readDataFile(dataFile,'output')
	#print "====>>> " , msrColNames

        featureNameToIndexMap_in,featureIndexToNameMap_in,colIdxToArrIdxMap_in = getColumnNameToIndexMappingFromFile(productionFile,inColNames)
        featureNameToIndexMap_msr,featureIndexToNameMap_msr,colIdxToArrIdxMap_msr = getColumnNameToIndexMappingFromFile(productionFile,msrColNames)
        featureNameToIndexMap_out,featureIndexToNameMap_out,colIdxToArrIdxMap_out = getColumnNameToIndexMappingFromFile(productionFile,outColNames)

	#print colIdxToArrIdxMap_msr
	#print featureIndexToNameMap_msr
        f = open(productionFile)
        lines = f.readlines()
	lineNum = 0
        for line in lines:
		inMap = {}
		out = {}
		msr = {}
		lineNum = lineNum + 1
                line = line.strip()
                if(line == ""):
                        continue
		if(lineNum == 1):
			continue
                fields = line.split("\t")
                colIdx = -1
                for field in fields:
			colIdx = colIdx + 1
                        if(colIdx in colIdxToArrIdxMap_in.keys()):
                                #print colIdx, "in"  
			        #arrIndex = colIdxToArrIdxMap_in[colIdx]
                                name = featureIndexToNameMap_in[colIdx]
                                inMap[name] = float(field)
                        if(colIdx in colIdxToArrIdxMap_msr.keys()):
                                #print colIdx, "msr"  
                                #arrIndex = colIdxToArrIdxMap_msr[colIdx]
                                name = featureIndexToNameMap_msr[colIdx]
                                msr[name] = float(field)
                        if(colIdx in colIdxToArrIdxMap_out.keys()):
                                #print colIdx, "out"  
                                #arrIndex = colIdxToArrIdxMap_out[colIdx]
                                name = featureIndexToNameMap_out[colIdx]
                                out[name] = float(field)

                fDpt = FeatureDataPoint(inMap)
		#print "\n\n --------------------\n", msr

		testProductionFile(anomalyDetectObject,productionFile,msr,out,fDpt,lineNum,isBasic)



if __name__ == "__main__" :
	referenceDataFile = sys.argv[1]
	productionDataFile = sys.argv[2]
	dumpDir = makeDumpDirectory(referenceDataFile)
        setGlobalObject("activeDumpDirectory",dumpDir)
	
	anoDetectEngine = anomalyDetectionEngine()
	anoDetectEngine.dumpDirectory = dumpDir
	anoDetectEngine.loadPerModuleObjects(referenceDataFile)

	#anoDetect = anomalyDetection()
	anoDetect = anoDetectEngine.getAnomalyDetectionObject(referenceDataFile)
	#anoDetect.errorProfPicklePath = getGlobalObject("activeDumpDirectory")
	#anoDetect.usefulFeaturePicklePath = "/home/mitra4/work/regression/selInput.cpkl"
	#anoDetect.dumpDirectory = getGlobalObject("activeDumpDirectory")
	print "\n**Results with ADVANCED technique\n---------------------------------------\n"
	
	createFeatureDataPointFromProductionFile(anoDetect,productionDataFile,False)
	
	print "\n**Results with BASIC technique\n-----------------------------------------------\n"
	
	createFeatureDataPointFromProductionFile(anoDetect,productionDataFile,True)
	#anoDetect.loadAnalysisFiles()
	#test_resultantError(anoDetect)
	#test_errorProfileMapLoad("/home/mitra4/work/regression/errMapDump.cpkl")
	#test_resultantError("/home/mitra4/work/regression/errMapDump.cpkl")
