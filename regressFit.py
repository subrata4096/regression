#!/usr/bin/python
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def doLinearRegression(inArr, targetArr):
	print "--------------------------------------------------------"     
	print "Linear"
	#first add a column of 1s so that it can be used as constant for linear regression
	idx = len(inArr[0])
        #inArrWithConst = np.insert(inArr,idx, 1, axis=1)
        inArrWithConst = inArr

	#clf = linear_model.Lasso(normalize=True)
        #clf = linear_model.LinearRegression( normalize=True)
	#clf = LinearRegression(fit_intercept=False) #fit_intercept=False is very important. prevents the LinearRegression object from working with x - x.mean(axis=0)
	clf = LinearRegression(fit_intercept=True)
        clf.fit (inArrWithConst,targetArr)
        print "R2 score: ",clf.score(inArrWithConst,targetArr)
        print "Coeff: ",clf.coef_
	#print clf.predict([[16, 3.615]])
        #print clf.predict([[8, 15]])
	#print clf.intercept_
        #print np.dot(inArrWithConst, clf.coef_) # reconstruct the output
        # show it on the plot
        #plt.plot(inArr[:,0], targetArr, label='true data')
        #plt.plot(testY, predSvr, 'co', label='SVR')
        #plt.plot(testY, predLog, 'mo', label='LogReg')
        #plt.legend()     
        #plt.show()

def doLinearRegWithCV(inArr,targetArr):
	print "--------------------------------------------------------"     
	print "LinearWithCV"
	#first add a column of 1s so that it can be used as constant for linear regression
        idx = len(inArr[0])
        #inArrWithConst = inArr
        inArrWithConst = np.insert(inArr,idx, 1, axis=1)
	clf = LinearRegression(fit_intercept=False)
	r = cross_validation.cross_val_score(clf, inArrWithConst, targetArr, cv=2)
	print r
	print clf.predict([16, 3.615])
        print clf.predict([8, 15])

def doRidgeWithCV(inArr,targetArr): 
	print "--------------------------------------------------------"     
	print "RidgeWithCV"     
	#alphas = np.linspace(0.1,0.1,10)  #np.linspace(0.1,20,100)
	#reg = RidgeCV(store_cv_values=True, alphas=None)
	#reg = RidgeCV()
	reg = Ridge(alpha=0.01,solver='cholesky')
	reg.fit(inArr,targetArr)
	#print reg.cv_values_
	print "R2 score: ",reg.score(inArr,targetArr)
	print "Coeff: ",reg.coef_
	#print reg.alpha_
	#print reg.predict([16, 3.615])
	#print reg.predict([8, 15])

def doPolyRegression(inArr, targetArr,tname):
	print "--------------------------------------------------------"     
	print "PolyRegression"     

	#inArr = np.array([[0, 0], [1,11], [2,12],[3,13]])
	#targetArr = np.array([0,3,6,12])
	#print inArr
	#print targetArr
	#poly = PolynomialFeatures(2)
	#print poly
	#polyInArr = poly.fit_transform(inArr)
	#print polyInArr
	#clf = LinearRegression(fit_intercept=False) #fit_intercept=False is very important. prevents the LinearRegression object from working with x - x.mean(axis=0)
        #clf.fit (polyInArr,targetArr)
        #print "R2 score: ", clf.score(polyInArr,targetArr)
	#clf.score_
        #print "Coeff: ", clf.coef_
	#x=[[16, 3.615]]
	#p = PolynomialFeatures(2).fit_transform(x)
	#print clf.predict(p)
	#x=[[8, 15]]
        #p = PolynomialFeatures(2).fit_transform(x)
        #print clf.predict(p)
        #print clf.predict([8, 15])
	
	polyReg = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))])
	#model = model.fit(inArr[:, np.newaxis], targetArr)
	polyReg.fit(inArr, targetArr)

	score = polyReg.score(inArr, targetArr)
	print "R2 score: ", score

	predicted = polyReg.predict(inArr)
	print predicted

	# Plot outputs
	fig = plt.figure(1, figsize=(8, 6))
	ax = Axes3D(fig, elev=-150, azim=110)
	ax.scatter(inArr[:,0],inArr[:,1],targetArr, c='b', marker='o')   #inArr[:,0] elements of first variable [0 1 2 3]
	ax.scatter(inArr[:,0],inArr[:,1],predicted, c='r',marker='^')   #inArr[:,0] elements of first variable [0 1 2 3]
	
	ax.set_title("Two input variables vs meassurement")
	ax.set_xlabel("numRanks")
	#ax.w_xaxis.set_ticklabels([])
	ax.set_ylabel("nx")
	#ax.w_yaxis.set_ticklabels([])
	ax.set_zlabel(tname)
	#ax.w_zaxis.set_ticklabels([])
	plt.show()
