#!/usr/bin/python

import cPickle
import pickle
# save the classifier
def dumpModel(tsvFName,target, model):
	import os.path
	locDir = os.path.dirname(tsvFName)
        onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
        #print onlyName,  target
        outName  = onlyName + "_" + target + ".pkl"
        outName2  = onlyName + "_" + target + ".cpkl"
        fullpath = os.path.join(locDir, outName)
        fullpath2 = os.path.join(locDir, outName2)
        print fullpath
	with open(fullpath, 'wb') as fid:
    		pickle.dump(model, fid)

        fid.close()    
	with open(fullpath2, 'wb') as fid:
                cPickle.dump(model, fid)

        fid.close()
	return fullpath
# load it again
def loadModel(pklFName):
	with open(pklFName, 'rb') as fid:
    		model_loaded = cPickle.load(fid)
	fid.close()
	return model_loaded

def dumpErrorDistributionProfileMap(errDistMap, dumpDir):
	picklepath = dumpDir + "/errMapDump.pkl"
	cPicklepath = dumpDir + "/errMapDump.cpkl"
	with open(picklepath, 'wb') as fid:
                pickle.dump(errDistMap, fid)

        fid.close()
        with open(cPicklepath, 'wb') as fid:
                cPickle.dump(errDistMap, fid)
        fid.close()
	return (picklepath,cPicklepath)

def dumpSelectedFeaturesMap(colNameToIndexMap,dumpDir):
	print "\n\n ------------------------- colNameToIndexMap", colNameToIndexMap
	picklepath = dumpDir + "/selectedFeatures.pkl"
	cPicklepath = dumpDir + "/selectedFeatures.cpkl"
        with open(picklepath, 'wb') as fid:
                pickle.dump(colNameToIndexMap, fid)

        fid.close()
        with open(cPicklepath, 'wb') as fid:
                cPickle.dump(colNameToIndexMap, fid)
        fid.close()
        return (picklepath,cPicklepath)

def loadSelectedFeaturesMap(dumpDir,isCPickle):
	pklFName = dumpDir + "/selectedFeatures"
	colNameToIndexMap = None
	if(isCPickle):
		pklFName = pklFName + ".cpkl"
                with open(pklFName, 'rb') as fid:
                        colNameToIndexMap = cPickle.load(fid)
                fid.close()
                return colNameToIndexMap
        else:
		pklFName = pklFName + ".pkl"
                with open(pklFName, 'rb') as fid:
                        colNameToIndexMap = pickle.load(fid)
                fid.close()
                return colNameToIndexMap

def loadErrorDistributionProfileMap(dumpDir, isCPickle):
	errProfileMap_loaded = None
	pklFName = dumpDir + "/errMapDump"
	if(isCPickle):
		pklFName = pklFName + ".cpkl"
		with open(pklFName, 'rb') as fid:
                	errProfileMap_loaded = cPickle.load(fid)
        	fid.close()
        	return errProfileMap_loaded
	else:
		pklFName = pklFName + ".pkl"
		with open(pklFName, 'rb') as fid:
                	errProfileMap_loaded = pickle.load(fid)
        	fid.close()
        	return errProfileMap_loaded

def dumpRegressorObjectDict(regressionObjectDict,dumpDir):
	picklepath = dumpDir + "/regressionObjectDict.pkl"
	cPicklepath = dumpDir + "/regressionObjectDict.cpkl"
        with open(picklepath, 'wb') as fid:
                pickle.dump(regressionObjectDict, fid)

        fid.close()
        with open(cPicklepath, 'wb') as fid:
                cPickle.dump(regressionObjectDict, fid)
        fid.close()
        return (picklepath,cPicklepath)	

def loadRegressorObjectDict(dumpDir, isCPickle):
        regressionObjectDict_loaded = None
        pklFName = dumpDir + "/regressionObjectDict"
        if(isCPickle):
                pklFName = pklFName + ".cpkl"
                with open(pklFName, 'rb') as fid:
                        regressionObjectDict_loaded = cPickle.load(fid)
                fid.close()
                return regressionObjectDict_loaded
        else:
                pklFName = pklFName + ".pkl"
                with open(pklFName, 'rb') as fid:
                        regressionObjectDict_loaded = pickle.load(fid)
                fid.close()
                return regressionObjectDict_loaded		
