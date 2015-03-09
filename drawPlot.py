#!/usr/bin/python
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
import pylab as plt
import os
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from fields import *

global inputColumnNames
global measuredColumnNames
global outputColumnNames


def doPlot(inArr,targetArr,in_index1,tname,reg=None):
        #return	        
	#xfullname = inputColumnNames[in_index1].split(':')
	#xfullname = getInputParameterNameFromColumnIndex(in_index1).split(':')
	xfullname = getInputParameterNameFromFeatureIndex(in_index1).split(':')
	xname = xfullname[len(xfullname) -1]
	print xname, " VS " , tname
        #fig = plt.figure(1, figsize=(8, 6))
        #ax = Axes(fig)
        #ax.scatter(inArr[:,in_index1],targetArr, c='b', marker='o',label='true data')   #inArr[:,0] elements of first variable [0 1 2 3]
        #ax.set_title("Input variables vs meassurement")
        #ax.set_xlabel(xname)
        #ax.w_xaxis.set_ticklabels([])
        #ax.set_ylabel(tname)
        #ax.w_yaxis.set_ticklabels([])
	#plt.legend() 
	
	plt.scatter(inArr[:,in_index1],targetArr, c='b', marker='o')   
	if(reg != None):
		predicted = reg.predict(inArr)
		#plt.plot(inArr[:,in_index1],predicted, c='r')
		plt.scatter(inArr[:,in_index1],predicted, c='r')
 
	plt.show()

def do3dPlot(inArr,targetArr,in_index1,in_index2,tname,reg=None):
	# Plot outputs
        #xfullname = inputColumnNames[in_index1].split(':')
        #xfullname = getInputParameterNameFromColumnIndex(in_index1).split(':')
        xfullname = getInputParameterNameFromFeatureIndex(in_index1).split(':')
        #yfullname = inputColumnNames[in_index2].split(':')
        yfullname = getInputParameterNameFromFeatureIndex(in_index2).split(':')

	xname = xfullname[len(xfullname) -1]
	yname = yfullname[len(yfullname) -1]

        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.scatter(inArr[:,in_index1],inArr[:,in_index2],targetArr, c='b', marker='o')   #inArr[:,0] elements of first variable [0 1 2 3]
        #ax.scatter(inArr[:,0],inArr[:,1],predicted, c='r',marker='^')   #inArr[:,0] elements of first variable [0 1 2 3]

	if(reg != None):
                predicted = reg.predict(inArr)
		ax.scatter(inArr[:,in_index1],inArr[:,in_index2],predicted, c='r',marker='^')   #inArr[:,0] elements of first variable [0 1 2 3]

	ax.set_title("Two input variables vs meassurement")
        ax.set_xlabel(xname)
        #ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel(yname)
        #ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel(tname)
        ax.w_zaxis.set_ticklabels([])
        plt.show()

def doHistogramPlot(dataSamples,targetName,featureName,doSave):
	#
	# first create a single histogram
	#
	mu = np.mean(dataSamples) 
	maxVal = max(dataSamples)
	minVal = min(dataSamples)
	if(float(abs(maxVal/mu)) > 100):
		dataSamples.remove(maxVal)
	if(float(abs(minVal/mu)) > 100):
		dataSamples.remove(minVal)
	#doSave = False
	doSave = True
	
	#print dataSamples
	mu = 0 
	sigma = np.std(dataSamples)
	
	tagetJustName = ""
	fields = targetName.split(":")
	targetJustName = fields[len(fields) - 1]
	isPapi = targetJustName.find("PAPI_")
	if(isPapi != -1):
		targetJustName = targetJustName[5:]
	
	appName = ""
	#appName = "Linpack"
	#appName = "Matrix Multiplication"
	#appName = "Sparse Matrix Vector Multiplication"
	#appName = "Black Scholes"
	#appName = "FFmeg"
	dumpDir = "/home/mitra4/work/regression/gold_histograms/"
	if(appName == "Sparse Matrix Vector Multiplication"):
		dumpDir = dumpDir + "SPARSE_MATRIX_MUL"
	if(appName == "Linpack"):
		dumpDir = dumpDir + "LINPACK"
	if(appName == "Matrix Multiplication"):
		dumpDir = dumpDir + "MATRIX_MUL"

	if(appName == "Sparse Matrix Vector Multiplication"):
		dataSamples = [x / 10.0 for x in dataSamples]
 
	fileName = "errHisto_" + getGlobalObject("baseModuleName") + ":" + targetName + "_" + featureName + ".png"
	fileName2 = "errHisto_" + getGlobalObject("baseModuleName") + ":" + targetName + "_" + featureName + "_tight.png"
        saveFileName = os.path.join(dumpDir,fileName)
        saveFileName2 = os.path.join(dumpDir,fileName2)

	#ttl = "Error histogram: \n" + targetName + " for " + featureName 
	ttl = appName + ": " + getGlobalObject("baseModuleName") + " - Observation: " + targetJustName

	bins = [-0.35,-0.25,-0.15,-0.05,0.05,0.15,0.25,0.35]
	#bins = np.arange(-0.3,-0.02,0.01) + np.array([-0.02,0.02]) +np.arange(0.02,0.3,0.01)
	bin1 = np.arange(-0.3,-0.02,0.02)
	#print bin1 
	bin2 = np.arange(0.02,0.3,0.02) 
	#print bin2 
	bins = np.concatenate([bin1,bin2])
	#print bins

	# the histogram of the data with histtype='step'
	#n, bins, patches = plt.hist(dataSamples, 50, histtype='stepfilled',normed=True,stacked=True)
	#n, bins, patches = plt.hist(dataSamples, 20, histtype='stepfilled',normed=True)
	#n, bins, patches = plt.hist(dataSamples, bins, histtype='bar')
	#n, bins, patches = plt.hist(dataSamples, bins, histtype='stepfilled',normed=True)
	#n, bins, patches = plt.hist(dataSamples, bins, histtype='stepfilled')
	n, bins, patches = plt.hist(dataSamples, 20, histtype='stepfilled')
	#n, bins, patches = plt.hist(dataSamples, 50, histtype='stepfilled')
	#n, bins, patches = plt.hist(dataSamples, 20, normed=1, histtype='stepfilled')
	#print "n = " , n
	#print "patches = " ,  patches	
	### TESTING START *******************
	#hist, bins = np.histogram(dataSamples,bins)
	# Create the histogram and normalize the counts to 1
	#sum_val = sum(hist)
	#max_val = max(hist)
	#hist = [ float(n)/sum_val for n in hist]
	# Plot the resulting histogram
	#center = (bins[:-1]+bins[1:])/2
	#center = 0
	#width = 1.0*(bins[1]-bins[0])
	#plt.bar(center, hist, align = 'center', width = width)
	#plt.show()

	## TESTING END **********************
	
	plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

	# add a line showing the expected distribution
	#y = plt.normpdf( bins, mu, sigma)
	#l = plt.plot(bins, y, 'k--', linewidth=1.5)
	plt.title(ttl)
	plt.xlabel('% Error')
	plt.ylabel('Frequency')
	#plt.yticks(np.arange(0,1,0.1))
	
	binLength = len(bins)
	fileName3 = "errHisto_" + getGlobalObject("baseModuleName") + ":" + targetName + "_" + featureName + "_bin" + str(binLength) +".eps"
	fileName4 = "errHisto_" + getGlobalObject("baseModuleName") + ":" + targetName + "_" + featureName + "_bin" + str(binLength) +".png"
        saveFileName3 = os.path.join(dumpDir,fileName3)
        saveFileName4 = os.path.join(dumpDir,fileName4)

	if(doSave == False):
        	plt.show()
		#plt.savefig(saveFileName)
		#plt.savefig(saveFileName2, bbox_inches='tight')
		plt.savefig(saveFileName3, bbox_inches='tight')
                plt.close()
	else:
		print "SaveFileName = " + saveFileName3
		#plt.savefig(saveFileName, bbox_inches='tight')
		#plt.savefig(saveFileName)
		#plt.savefig(saveFileName2, bbox_inches='tight')
		plt.savefig(saveFileName3, format='eps',dpi=800,bbox_inches='tight')
		plt.savefig(saveFileName4, format='png',dpi=800,bbox_inches='tight')
                plt.close()

def drawErrorDistPlotWithFittedCurve(errorSamples,Distances,targetName,featureName,curve,doSave):
	doSave = True
        #distList = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	sortedDist = np.sort(Distances)
	maxDist = sortedDist[len(sortedDist) - 1] 
	minDist = sortedDist[0]
	distList = np.linspace(minDist,maxDist,10)  # make 10 equally spaced point between minDist and maxDist in the obs 
	errList = []
	#dumpDir = "/home/mitra4/work/regression/gold_plots/LINPACK_PLOTS" 
	dumpDir = "/home/mitra4/work/regression/gold_plots/MATRIX_MUL" 
	for d in distList:
		err = curve.predict(d)
		if(d < 1):
			err = 0.0
		errList.append(abs(err))

	x = distList
	y = errList
	ttl = "Absolute error against distance: \n" + targetName + " for " + featureName
        print ttl
        #plt.plot(Distances,errorSamples)
        #plt.scatter(x,y, c='b', marker='o')
	# to make it smooth
        [x, y] = zip(*sorted(zip(x, y), key=lambda x: x[0]))
        plt.plot(x,y)
        plt.title(ttl)
	fileName = "errVSdist_" + getGlobalObject("baseModuleName") + ":" + targetName + "_" + featureName + "_predicted.png"
        saveFileName = os.path.join(dumpDir,fileName)
	if(doSave == False):
                plt.show()
        else:
                print "SaveFileName = " + saveFileName
                plt.savefig(saveFileName, bbox_inches='tight')
		plt.close()
                #plt.savefig(saveFileName)
	
def drawErrorDistPlot(errorSamples,Distances,targetName,featureName,doSave):
	doSave = True
	#return
	#dumpDir = "/home/mitra4/work/regression/gold_plots/LINPACK_PLOTS"
	dumpDir = "/home/mitra4/work/regression/gold_plots/MATRIX_MUL"
	xDict = {}
	yList = []
        x = []
	y = []
	#make a dictionary to capture all the error values for a particular distance (ROUNDED upto 3-decimal places)
	i =0
	for d in Distances:
		#print d
		distAbs = abs(d)	
		#round up to 3 decimal places
		distKey = round(distAbs,3)
		e = abs(errorSamples[i])
		if distKey in xDict.keys():
			xDict[distKey].append(e)
		else:
			xDict[distKey] = []
			xDict[distKey].append(e)

		i = i + 1
	#end for
	
	for key in xDict.keys():
		meanErr  = np.mean(xDict[key])
                #print "distance=",key," error list=",xDict[key]			
		x.append(key)
		y.append(meanErr)
			

	#for e in errorSamples:
	#	y.append(abs(e))	
	ttl = "Error with distance: \n" + targetName + " for " + featureName
	print ttl 
	#plt.plot(Distances,errorSamples)
	# to make it smooth
	[x, y] = zip(*sorted(zip(x, y), key=lambda x: x[0]))
	plt.plot(x,y)
	#plt.scatter(x,y, c='b', marker='o')
	plt.title(ttl)
	fileName = "errVSdist_" + getGlobalObject("baseModuleName") + ":" + targetName + "_" + featureName + "_actual.png"
        saveFileName = os.path.join(dumpDir,fileName)
        if(doSave == False):
                plt.show()
        else:
                print "SaveFileName = " + saveFileName
                plt.savefig(saveFileName, bbox_inches='tight')
		plt.close()
                #plt.savefig(saveFileName)
        

def plotFromFile(fileName,xColNumber,yColNumber):
	indexes = [xColNumber]
	xdata = np.loadtxt(fileName, dtype=float,delimiter='\t', usecols=indexes, converters=None,skiprows=1)
	print xdata
	indexes = [yColNumber]
	ydata = np.loadtxt(fileName, dtype=float,delimiter='\t', usecols=indexes, converters=None,skiprows=1)

	d = np.genfromtxt(fileName, dtype=str,delimiter='\t')

        names = d[0]
	xName = names[xColNumber]
	yName = names[yColNumber]

	ttl = xName + " VS " + yName
	plt.scatter(xdata,ydata, c='b', marker='o')
	#plt.plot(xdata,ydata, c='r')
	
	plt.title(ttl)
        plt.show()

if __name__ == "__main__":
	dataFile = sys.argv[1]
	xColNum = int(sys.argv[2])
	yColNum = int(sys.argv[3])
	plotFromFile(dataFile,xColNum,yColNum)

