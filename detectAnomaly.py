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

global inputColumnNames
global measuredColumnNames
global outputColumnNames

global regressionDict


def check_anomaly(production_inArr, targetArr,tname):
        reg = regressionDict[tname]
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
        i = 0
        for targetArr in measuredArr:
                t = measuredColumnNames[i]
                #check_anomaly(inArr,targetArr,t)
                check_score(inArr,targetArr,t)
                i = i + 1

        i = 0
        for targetArr in outArr:
                t = outputColumnNames[i]
                #check_anomaly(inArr,targetArr,t)
                check_score(inArr,targetArr,t)
                i = i + 1

def check_score(production_inArr, targetArr,tname):
	reg = regressionDict[tname]
        if(reg == None):
                print "target", tname, " can not be predicted by the captured inputs"
                return

	print "Production R2 score: ",tname , "= " , reg.score(production_inArr, targetArr)	
