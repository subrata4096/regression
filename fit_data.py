#!/usr/bin/python
from sklearn import linear_model
import numpy as np
import sys

def readInputCombinationFile(fName):
	print fName
	inputs = np.loadtxt(fName, dtype=float,delimiter='\t',converters=None,skiprows=1)
	#print inputs
	return inputs


def readDataFile(fName):
	print fName
	realdata = np.loadtxt(fName, dtype=float,delimiter='\t',converters=None,skiprows=1)
	data = np.transpose( realdata)
	#print data
	return data
	

def scikit_scripts(inArr,dataArr):
	from sklearn import linear_model
	a = inArr
	b = dataArr[2]
	c = b
	#print a
	#print c
	#clf = linear_model.LinearRegression()
	clf = linear_model.Lasso(normalize=True)
	clf.fit (inArr,c)
	print clf.score(inArr,c)
	#print clf.coef_
      
        from sklearn import svm	
	p = svm.SVR()
	#p = svm.SVR(kernel='linear')
	p.fit(a,c)
	print p.score(inArr,c)
	#print p.coef_


	from sklearn.preprocessing import PolynomialFeatures
       	from sklearn.linear_model import LinearRegression
	from sklearn.pipeline import Pipeline
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import Ridge
	model = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=True))])
	model = model.named_steps['linear'].fit(a,c)
	#model = model.named_steps['poly'].fit(a,c)
        #model = make_pipeline(PolynomialFeatures(degree=2), Ridge())
	#print model.score(inArr,c)
	#print model.coef_

        rdg = Ridge(alpha=1.0,normalize=True)
	rdg.fit(a,c)

	x = inArr[14]
	print x
	y1 = clf.predict(x)
	y2 = p.predict(x)
	y3 = model.predict(x)
	y4 = rdg.predict(x)

	print y1
	print y2
	print y3
	print y4


       	from sklearn import svm, grid_search
	parameters = {'alpha':[1,2,3,5,10]}	
        r = linear_model.Ridge()
	mm = grid_search.GridSearchCV(r, parameters,cv=3)
	mm.fit(a,c)
	y101 = mm.predict(x)
	print y101
	print mm.best_score_
	

#def skll_scripts(inArr,dataArr):
#	import skll
#	learner = skll.Learner('LinearRegression')
	









if __name__ == "__main__":
	inputCombinationFile = sys.argv[1]
	measureDataFile = sys.argv[2]
	
	inputArr = readInputCombinationFile(inputCombinationFile)
	dataArr = readDataFile(measureDataFile)
	scikit_scripts(inputArr,dataArr)
