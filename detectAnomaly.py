#!/usr/bin/python
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
import sys
import fnmatch
from scipy import stats
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


#def createFeatureDataPointFromProductionFile(productionFile):
#	inColumnNames,msrColumnNames,outColumnNames = parseFields(productionFile)
#
#	#inputDataArr = readDataFile(dataFile,'input')
#	#measuredDataArr = readDataFile(dataFile,'measured')
#	#outputDataArr = readDataFile(dataFile,'output')
#
#	featureNameToIndexMap_in,featureIndexToNameMap_in,colIdxToArrIdxMap_in = getColumnNameToIndexMappingFromFile(productionFile,inColNames)	
#	featureNameToIndexMap_msr,featureIndexToNameMap_msr,colIdxToArrIdxMap_msr = getColumnNameToIndexMappingFromFile(productionFile,msrColNames)	
#	featureNameToIndexMap_out,featureIndexToNameMap_out,colIdxToArrIdxMap_out = getColumnNameToIndexMappingFromFile(productionFile,outColNames)	


#	f = open(productionFile):
#		lines = f.readLines()
#	for line in lines:
#		line = line.strip()
#		if(line == ""):
#			continue
#		fields = line.split("\t")
#		colIdx = 0
#		for field in fields:
#			if(colIdx in colIdxToArrIdxMap_in.keys()):
#				arrIndex = colIdxToArrIdxMap_in[colIdx]
#				name = featureIndexToNameMap_in[arrIndex]
#				inMap[name] = float(field)
#			if(colIdx in colIdxToArrIdxMap_msr.keys()):
#                                arrIndex = colIdxToArrIdxMap_msr[colIdx]
#                                name = featureIndexToNameMap_out[arrIndex]
#                                msr[name] = float(field)
#			if(colIdx in colIdxToArrIdxMap_out.keys()):
#                                arrIndex = colIdxToArrIdxMap_out[colIdx]
#                                name = featureIndexToNameMap_out[arrIndex]
#                                out[name] = float(field)

#		fDpt = FeatureDataPoint(inMap)
		


class anomalyDetection:
	def __init__(self):
		#self.errorProfPicklePath = ''
		#self.usefulFeaturePicklePath = ''
		self.fixedThreshold = None
		self.dumpDirectory = ''
		self.errProfileMap = None
		self.selectedFeatureMap = None
		self.selectedFeatureSortedByInIndex = None
		self.regressionObjectDict = None
		self.isValid = False
		self.goodTargetMap = None
		self.targetErrorMap = None

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

		self.goodTargetMap = loadGoodTargetMap(self.dumpDirectory,True,tsvFileName)
		if(self.goodTargetMap == None):
                        print tsvFileName," Good Target Map could not be loaded"
                        self.isValid = False
                        return

		#print self.goodTargetMap

		self.targetErrorMap = loadTargetErrMap(self.dumpDirectory,True,tsvFileName)
                if(self.targetErrorMap == None):
                        print tsvFileName," Target Error Map could not be loaded"
                        self.isValid = False
                        return
		
		self.isValid = True
			
	def getPredictionErrorEstimation(self, targetName, productionPt):
		dataPtWithSelectedFeature = errorDatastructure.getSelectedFeaturePtFromProductionDataPoint(productionPt,self.selectedFeatureMap)
		rmsErr,errPosBias,errNegBias = getResultantErrorFromFeatureErrorsForATargetAtADatapoint(targetName,dataPtWithSelectedFeature,self.errProfileMap)

        	#print "Percentage of overall error = ", rmsErr, errPosBias,errNegBias

		return rmsErr,errPosBias,errNegBias

	#Based on our prediction and error estimation on top of that we calculate valid target value range
	#This value is based on on input feature value provided for production settings
	#If observed value falls outside this range: it will be an anomaly
	def getValidRangeOfTargetValue(self, targetName, productionPt):
		dataPtWithSelectedFeature = errorDatastructure.getSelectedFeaturePtFromProductionDataPoint(productionPt,self.selectedFeatureMap)
		
		rmsErr,errPosBias,errNegBias = getResultantErrorFromFeatureErrorsForATargetAtADatapoint(targetName,dataPtWithSelectedFeature,self.errProfileMap)
		inArr = errorDatastructure.getSelectedInputArrFromSelectedDataPoint(dataPtWithSelectedFeature,self.selectedFeatureSortedByInIndex)	
		predictedVal = 0
		#print "TEST : Selected input array is: ", inArr 
		#d = [100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000]
		#drawErrorDistPlotWithFittedCurve([],d,targetName,"B",self.regressionObjectDict[targetName],False)
		predictedVal = self.regressionObjectDict[targetName].predict(inArr)	
		#try:
		#except OverflowError:
		#	print "ERRROR           #######################################" 
		
		predictedValueErrorAdjusted = (predictedVal,predictedVal,predictedVal)


		errorVal = predictedVal * rmsErr
		
		#print "predictedVal=", predictedVal

		#if(errPosBias):
		#	predictedValueErrorAdjusted = (predictedVal, predictedVal + abs(errorVal))
		#elif(errNegBias):
		#	predictedValueErrorAdjusted = (predictedVal - abs(errorVal), predictedVal)
		#else:
		#	predictedValueErrorAdjusted = (predictedVal - abs(errorVal), predictedVal + abs(errorVal))
		predictedValueErrorAdjusted = (abs(predictedVal) - abs(errorVal), abs(predictedVal), abs(predictedVal) + abs(errorVal),abs(rmsErr))

		#print "Predicted val = ", predictedVal, " Valid range of value for = ", targetName, "  is = ", predictedValueErrorAdjusted
		return predictedValueErrorAdjusted

	def getValidRangeOfTargetValueBasic(self, targetName, productionPt):
		dataPtWithSelectedFeature = errorDatastructure.getSelectedFeaturePtFromProductionDataPoint(productionPt,self.selectedFeatureMap)
		inArr = errorDatastructure.getSelectedInputArrFromSelectedDataPoint(dataPtWithSelectedFeature,self.selectedFeatureSortedByInIndex)	
		predictedVal = self.regressionObjectDict[targetName].predict(inArr)	
		
		#print "predictedVal=", predictedVal

		#errorVal = predictedVal * 0.5
		errorVal = predictedVal * self.fixedThreshold

		predictedValueErrorAdjusted = (predictedVal - abs(errorVal), predictedVal, predictedVal + abs(errorVal))

		return predictedValueErrorAdjusted


def getProbabilityOfAnError(errSamples, deviation):
	meanOfSamples = np.mean(errSamples)
	errSamples = errSamples - meanOfSamples # shift the samples to have zero mean 
	meanOfSamples = np.mean(errSamples)
	#print errSamples
	#print "meanOfSamples= ",meanOfSamples, " deviation= ", deviation
	prob = 0
	kernel = stats.gaussian_kde(errSamples)
	#val = kernel.integrate_gaussian(np.mean(errSamples),np.std(errSamples))
	negDev = -1.0*deviation
	posDev = 1.0*deviation
	anomalyProb = kernel.integrate_box_1d(negDev, posDev)
	#val3 = kernel.integrate_box_1d(-0.0001, 0.0001)
	#val4 = kernel.integrate_box_1d(-1.0, 1.0)
	#print anomalyProb,val4
	return anomalyProb

	#if(deviation == 0):
	#	return 0.0
	#elif(deviation < meanOfSamples):
	#	countA = 0
	#	countB = 0
	#	for e in errSamples:
	#		if(e < meanOfSamples):
	#			countB = countB + 1
	#		if(e < deviation):
	#			countA = countA + 1
	#	#end for
	#	prob = 1 - float(countA/countB)
	#	return prob
	#elif(deviation >= meanOfSamples):
	#	countA = 0
        #        countB = 0
        #        for e in errSamples:
        #                if(e > meanOfSamples):
        #                        countB = countB + 1
        #                if(e > deviation):
        #                        countA = countA + 1
                #end for
        #        prob = 1 - float(countA/countB)
        #        return prob



def getStackFromFileLocation(dumpDir,fileLoc):
	baseName = os.path.basename(fileLoc)
	return baseName
	idx = fileLoc.find(dumpDir)
	lengthStr = len(dumpDir)
	stackStr = fileLoc[idx+lengthStr+1 : ]
	print "stack string is:" + stackStr
	return stackStr
class anomalyDetectionEngine:
	def __init__(self):
		self.dumpDirectory = ""
		self.anomalyDetectionPerModuleObjectMap = {}
	def loadPerModuleObjects(self,oneParticularFile=""):
                if(self.dumpDirectory == ""):
                        print "\nERROR: Please set dumpDirectory on anomalyDetectionEngine object\n"
                        exit(0)

                dumpedFiles = []
                dumpedFiles = searchForDumpedFiles(self.dumpDirectory, "*.tsv")
		if(oneParticularFile != ""):
			for afile in dumpedFiles:
				if(afile.find(oneParticularFile) != -1):
					dumpedFiles = [afile]
                for afile in dumpedFiles:
			anoDetect = anomalyDetection()
			anoDetect.dumpDirectory = self.dumpDirectory
                        anoDetect.loadAnalysisFiles(afile)
			if(anoDetect.isValid):
				stackStr = getStackFromFileLocation(self.dumpDirectory,afile)
				self.anomalyDetectionPerModuleObjectMap[stackStr] = anoDetect
		print self.anomalyDetectionPerModuleObjectMap

	def getAnomalyDetectionObject(self,filename):
		#print filename
		baseName = os.path.basename(filename)
		if baseName in self.anomalyDetectionPerModuleObjectMap.keys():
			return self.anomalyDetectionPerModuleObjectMap[baseName]
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
