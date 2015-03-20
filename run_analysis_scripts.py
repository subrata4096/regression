#!/usr/bin/python

import os
import sys

if __name__ == "__main__" :
	targetTSVDataFile = sys.argv[1]
	logAnalyze = targetTSVDataFile + "_analyze.log"
	logErrHisto = targetTSVDataFile + "_errHisto.log"
	logErrProfile = targetTSVDataFile + "_errProfile.log"

	com1 = "python analyze.py " + "\'" + targetTSVDataFile + "\'"
	#com1 = "nohup python analyze.py " + targetTSVDataFile + " > " + logAnalyze  + " &"
	
	com2 = "python errorHistogramAnalysis.py " + "\'" + targetTSVDataFile + "\'"
	#com2 = "nohup python errorHistogramAnalysis.py " + targetTSVDataFile + " > " + logErrHisto  + " &"
	
	com3 = "python errorAnalysisControlled.py " + "\'" + targetTSVDataFile + "\'"
	#com3 = "nohup python errorAnalysisControlled " + targetTSVDataFile + " > " + logErrProfile + " &"

	#print com1
	#os.system(com1)

	#print com2
	#os.system(com2)

	print com3
	os.system(com3)
