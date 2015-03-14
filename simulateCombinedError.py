#!/usr/bin/python
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
import os.path
import sys
import math
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def sigNum(val):
	if(val == 0):
		return 0
	if(val > 0):
		return 1
	if(val < 0):
		return -1




def do2DSimulation():
	
	fig = plt.figure()	
	#ax = fig.gca(projection='2d')
	ax = fig.add_subplot(111)
	mCoeffList = np.linspace(0,1,200)	
	#mCoeffList = [1]	
	E1 = 1.0
	E2 = 1.0
	err1List = []
	err2List = []
        print mCoeffList	
	for mCoeff in mCoeffList:
		n = float(1 + sigNum(mCoeff))
		m = float(1 + mCoeff)
		c1 = float((1/n)*(abs(E1) + abs(mCoeff*E2)))
		c1P = float((1/m)*(abs(E1) + abs(mCoeff*E2)))

		c2 = float(abs((1 - mCoeff)*E2))
		
		#print "n: ", n, " c1: ", c1, "c2: ", c2

		#err1 = math.sqrt( math.pow(c1,2) + math.pow(c2,2)) 
		err2 = math.sqrt( math.pow(c1P,2) + math.pow(c2,2)) 
		#err1List.append(err1)
		err2List.append(err2)
         	
	#ax.grid(color='black', linestyle='-', linewidth=1)
	ax.set_xticks([0,0.2,0.8,1.0])
	ax.set_yticks([1.0,1.2,1.4])
	ax.set_xlabel("$m_{f_{1}f_{2}}$",fontsize=28)
	ax.set_ylabel("Overall error",fontsize=24)
	for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(16)
	for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16) 
	#plt.plot(mCoeffList,err1List,c='b')
	#plt.plot(mCoeffList,err2List,c='b')
	ax.plot(mCoeffList,err2List,c='b')
	plt.figtext(0.25, 0.8, "$|E_{f_{1}}| = |E_{f_{2}}| = 1.0$",fontsize=24)
	#plt.title("Variation of overall error with MIC between 2 error components",fontsize=24)
	plt.savefig("simu2d.eps",format="eps",dpi=800,bbox_inches="tight")	
	plt.show()
	plt.close()

def calculateRMSError3Comps(mCoeff1,mCoeff2):
        E1 = 1.0
        E2 = 1.0
        E3 = 1.0
        #c1 = float((1/n)*(abs(E1) + abs(mCoeff1*E2) + abs(mCoeff2*E3)))
        c1 = float((abs(E1) + abs(mCoeff1*E2) + abs(mCoeff2*E3)))
        c2 = float(abs(E2))
        c3 = float(abs(E3))
        #print "n: ", n, " c1: ", c1, "c2: ", c2
        #err = math.sqrt( math.pow(c1,2) + math.pow(c2,2) + math.pow(c3,2))
        err = math.sqrt( math.pow(c1,2) + math.pow(c2,2) + math.pow(c3,2))
        return err
	
def calculateError3Comps(mCoeff1,mCoeff2):
	E1 = 1.0
        E2 = 1.0
        E3 = 1.0
	#n = float(1 + sigNum(mCoeff1) + sigNum(mCoeff2))
        m = float(1 + mCoeff1 + mCoeff2)
        #c1 = float((1/n)*(abs(E1) + abs(mCoeff1*E2) + abs(mCoeff2*E3)))
        c1P = float((1/m)*(abs(E1) + abs(mCoeff1*E2) + abs(mCoeff2*E3)))
        c2 = float(abs((1 - mCoeff1)*E2))
	c3 = float(abs((1 - mCoeff2)*E3))
        #print "n: ", n, " c1: ", c1, "c2: ", c2
        #err = math.sqrt( math.pow(c1,2) + math.pow(c2,2) + math.pow(c3,2))
        err = math.sqrt( math.pow(c1P,2) + math.pow(c2,2) + math.pow(c3,2))
	return err
	
def do3DSimulation():

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	mCoeff1List = np.linspace(0,1,12)
	mCoeff2List = np.linspace(0,1,12)
        #mCoeffList = [1]       
        err1List = []
        err2List = []
        #print mCoeffList

	X, Y = np.meshgrid(mCoeff1List, mCoeff2List)
	z1s = np.array([calculateError3Comps(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	z2s = np.array([calculateRMSError3Comps(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	#R = np.sqrt(X**2 + Y**2)
	#Z = np.sin(R)
	Z1 = z1s.reshape(X.shape)
	Z2 = z2s.reshape(X.shape)
	#surf = ax.plot_surface(X, Y, Z1)
	#surf = ax.plot_surface(X, Y, Z2)
	#surf = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.afmhot,linewidth=0, antialiased=False)
	surf = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.afmhot)
	ax.set_zlim(0, 1.8)
	ax.view_init(elev=25,azim=-25)

	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xticks([0,0.2,0.8,1.0])
	ax.set_yticks([0,0.2,0.8,1.0])
	ax.set_zticks([0,0.5,1.0,1.8])
	#ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
	#fig.colorbar(surf, shrink=0.5, aspect=5)

	for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(15) 
	for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(15)
	for tick in ax.zaxis.get_major_ticks():
                tick.label.set_fontsize(15)
	ax.grid(color='black',linestyle='-', linewidth=1)

	ax.set_xlabel("$m_{f_{1}f_{2}}$",fontsize=28)
	ax.set_ylabel("$m_{f_{1}f_{3}}$",fontsize=28)
	ax.set_zlabel("Overall error",fontsize=24)
	plt.figtext(0.25, 0.8, "$|E_{f_{1}}| = |E_{f_{2}}| = |E_{f_{3}}| = 1.0$",fontsize=24)
	#plt.title("Variation of overall error with MIC between 3 error components",fontsize=24)
	plt.savefig("simu3d.eps",format="eps",dpi=800,bbox_inches="tight")	
        	
        plt.show()	
	
if __name__ == "__main__":
	do2DSimulation()
	do3DSimulation()
