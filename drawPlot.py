#!/usr/bin/python
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from fields import *

global inputColumnNames
global measuredColumnNames
global outputColumnNames

def doPlot(inArr,targetArr,in_index1,tname,reg=None):
        #return	        
	xfullname = inputColumnNames[in_index1].split(':')
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
        xfullname = inputColumnNames[in_index1].split(':')
        yfullname = inputColumnNames[in_index2].split(':')

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