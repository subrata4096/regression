#!/usr/bin/python
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
import sys
import fnmatch
from regressFit import *
from micAnalysis import *
from drawPlot import *
from fields import *
from pickleDump import *
import errorDatastructure
from errorAnalysis import *

#global inputColumnNames
#global measuredColumnNames
#global outputColumnNames

#global regressionDict

def searchForDumpedFiles(dirLoc,search_for_extension):
        matches = []
	print "directory:" + dirLoc
        for root, dirnames, filenames in os.walk(dirLoc):
                for filename in fnmatch.filter(filenames, search_for_extension):
                        matches.append(os.path.join(root, filename))
        print matches
        return matches


class anomalyDetection:
	def __init__(self):
		#self.errorProfPicklePath = ''
		#self.usefulFeaturePicklePath = ''
		self.dumpDirectory = ''
		self.errProfileMap = None
		self.selectedFeatureMap = None
		self.selectedFeatureSortedByInIndex = None
		self.regressionObjectDict = None
		self.isValid = False

	def loadAnalysisFiles(self,tsvFileName):
		#self.errProfileMap = loadErrorDistributionProfileMap(self.errorProfPicklePath,True)
		self.errProfileMap = loadErrorDistributionProfileMap(self.dumpDirectory,True,tsvFileName)
		#self.selectedFeatureMap = loadSelectedFeaturesMap(self.usefulFeaturePicklePath,True)
		#print self.errProfileMap["o3"]
		#errorDatastructure.printErrorDistributionProfileMapForTargetAndFeatureMap(self.errProfileMap)
		#print self.selectedFeatureMap

		if(self.errProfileMap == None):
			print tsvFileName," Error Profile Map could not be loaded"
			self.isValid = False
			return

		self.selectedFeatureMap = loadSelectedFeaturesMap(self.dumpDirectory,True,tsvFileName)
		if(self.selectedFeatureMap == None):
			print tsvFileName," Selected Feature Map could not be loaded"
			self.isValid = False
			return
		
		self.selectedFeatureSortedByInIndex = getSortedTupleFromDictionary(self.selectedFeatureMap)
		
		self.regressionObjectDict = loadRegressorObjectDict(self.dumpDirectory,True,tsvFileName)
		if(self.regressionObjectDict == None):
                        print tsvFileName," Regression Object Dict could not be loaded"
			self.isValid = False
			return

		
		self.isValid = True
			
	def getPredictionErrorEstimation(self, targetName, productionPt):
		dataPtWithSelectedFeature = errorDatastructure.getSelectedFeaturePtFromProductionDataPoint(productionPt,self.selectedFeatureMap)
		rmsErr,errPosBias,errNegBias = getResultantErrorFromFeatureErrorsForATargetAtADatapoint(targetName,dataPtWithSelectedFeature,self.errProfileMap)

        	print "Percentage of overall error = ", rmsErr, errPosBias,errNegBias

		return rmsErr,errPosBias,errNegBias

	#Based on our prediction and error estimation on top of that we calculate valid target value range
	#This value is based on on input feature value provided for production settings
	#If observed value falls outside this range: it will be an anomaly
	def getValidRangeOfTargetValue(self, targetName, productionPt):
		dataPtWithSelectedFeature = errorDatastructure.getSelectedFeaturePtFromProductionDataPoint(productionPt,self.selectedFeatureMap)
		
		rmsErr,errPosBias,errNegBias = getResultantErrorFromFeatureErrorsForATargetAtADatapoint(targetName,dataPtWithSelectedFeature,self.errProfileMap)
		inArr = errorDatastructure.getSelectedInputArrFromSelectedDataPoint(dataPtWithSelectedFeature,self.selectedFeatureSortedByInIndex)	
		
		print "TEST : Selected input array is: ", inArr 
		predictedVal = self.regressionObjectDict[targetName].predict(inArr)	
		
		predictedValueErrorAdjusted = (predictedVal,predictedVal)

		errorVal = predictedVal * rmsErr

		if(errPosBias):
			predictedValueErrorAdjusted = (predictedVal, predictedVal + errorVal)
		elif(errNegBias):
			predictedValueErrorAdjusted = (predictedVal, predictedVal - errorVal)
		else:
			predictedValueErrorAdjusted = (predictedVal - errorVal, predictedVal + errorVal)

		print "Predicted val = ", predictedVal, " Valid range of value for = ", targetName, "  is = ", predictedValueErrorAdjusted
		return predictedValueErrorAdjusted


def getStackFromFileLocation(dumpDir,fileLoc):
	idx = fileLoc.find(dumpDir)
	lengthStr = len(dumpDir)
	stackStr = fileLoc[idx+lengthStr+1 : ]
	print "stack string is:" + stackStr
	return stackStr
class anomalyDetectionEngine:
	def __init__(self):
		self.dumpDirectory = ""
		self.anomalyDetectionPerModuleObjectMap = {}
	def loadPerModuleObjects(self):
                if(self.dumpDirectory == ""):
                        print "\nERROR: Please set dumpDirectory on anomalyDetectionEngine object\n"
                        exit(0)

                #dumpedFiles = searchForDumpedFiles(self.dumpDirectory, "*.cpkl")
                dumpedFiles = searchForDumpedFiles(self.dumpDirectory, "*.tsv")
                for file in dumpedFiles:
			anoDetect = anomalyDetection()
			anoDetect.dumpDirectory = self.dumpDirectory
                        anoDetect.loadAnalysisFiles(file)
			if(anoDetect.isValid):
				stackStr = getStackFromFileLocation(self.dumpDirectory,file)
				self.anomalyDetectionPerModuleObjectMap[stackStr] = anoDetect
		print self.anomalyDetectionPerModuleObjectMap

	def getAnomalyDetectionObject(self,filename):
		#print filename
		if filename in self.anomalyDetectionPerModuleObjectMap.keys():
			return self.anomalyDetectionPerModuleObjectMap[filename]
		else:
			return None	


def check_anomaly(production_inArr, targetArr,tname):
        reg = getGlobalObject("regressionDict")[tname]
        if(reg == None):
                print "target", tname, " can not be predicted by the captured inputs"
                return
        #print "Input arr: ", production_inArr
        predicted = reg.predict(production_inArr)
        #print tname, " R2 score: ",reg.score(production_inArr, targetArr)
        #print tname, " prediction: ",reg.predict(production_inArr)
        #print tname, " Actual: ", targetArr
        error = (targetArr[0] - predicted[0])*1.0/float(targetArr[0])
        print "Percentage error: " , error

def anomaly_detection(inArr, measuredArr, outArr):
	msrdCols = getGlobalObject("measuredColumnNames")
	outputCols = getGlobalObject("outputColumnNames")
        i = 0
        for targetArr in measuredArr:
                #t = measuredColumnNames[i]
                t = msrdCols[i]
                #check_anomaly(inArr,targetArr,t)
                check_score(inArr,targetArr,t)
                i = i + 1

        i = 0
        for targetArr in outArr:
                #t = outputColumnNames[i]
                t = outputCols[i]
                #check_anomaly(inArr,targetArr,t)
                check_score(inArr,targetArr,t)
                i = i + 1

def check_score(production_inArr, targetArr,tname):
	reg = getGlobalObject("regressionDict")[tname]
        if(reg == None):
                print "target", tname, " can not be predicted by the captured inputs"
                return

	print "Production R2 score: ",tname , "= " , reg.score(production_inArr, targetArr)	

#if __name__ == "__main__":
#	fValMap = {}
#	fDtPt = errorDatastructure.FeatureDataPoint(fValMap)
#	print fDtPt.featureNameValueMap
#	errorDatastructure.getSelectedFeaturePtFromProductionDataPoinr(fDtPt,fValMap)
