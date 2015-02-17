#!/usr/bin/python
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
import sys
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

class anomalyDetection:
	def __init__(self):
		self.errorProfPicklePath = ''
		self.usefulFeaturePicklePath = ''
		self.errProfileMap = None
		self.selectedFeatureMap = None

	def loadAnalysisFiles(self):
		self.errProfileMap = loadErrorDistributionProfileMap(self.errorProfPicklePath,True)
		self.selectedFeatureMap = loadSelectedFeaturesMap(self.usefulFeaturePicklePath,True)
		print self.errProfileMap
		errorDatastructure.printErrorDistributionProfileMapForTargetAndFeatureMap(self.selectedFeatureMap)
		print self.selectedFeatureMap
		if(self.errProfileMap == None):
			print "Error Profile Map could not be loaded"
		if(self.selectedFeatureMap == None):
			print "Selected Feature Map could not be loaded"
			
	def getPredictionErrorEstimation(self, targetName, productionPt):
		dataPtWithSelectedFeature = errorDatastructure.getSelectedFeaturePtFromProductionDataPoint(productionPt,self.selectedFeatureMap)
		rmsErr,errPostibeBias,errMegBias = getResultantErrorFromFeatureErrorsForATargetAtADatapoint(targetName,dataPtWithSelectedFeature,self.errProfileMap)

        	print rmsErr, errPostibeBias,errPostibeBias

		return rmsErr,errPostibeBias,errMegBias
		
			

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
