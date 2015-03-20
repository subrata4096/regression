#!/usr/bin/python

import cPickle
import pickle
import os

# save the classifier
def dumpModel(tsvFName,target, model):
	import os.path
	locDir = os.path.dirname(tsvFName)
        onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
        #print onlyName,  target
        #outName  = onlyName + "_" + target + ".pkl"
        outName2  = onlyName + "_" + target + ".cpkl"
        #fullpath = os.path.join(locDir, outName)
        fullpath2 = os.path.join(locDir, outName2)
        print fullpath2
	#with open(fullpath, 'wb') as fid:
    	#	pickle.dump(model, fid)

        #fid.close()    
	with open(fullpath2, 'wb') as fid:
                cPickle.dump(model, fid)

        fid.close()
	return fullpath2
# load it again
def loadModel(pklFName):
	with open(pklFName, 'rb') as fid:
    		model_loaded = cPickle.load(fid)
	fid.close()
	return model_loaded

def dumpErrorDistributionProfileMap(errDistMap, dumpDir,tsvFName):
	onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
        filename = onlyName + "_errMapDump"	
	picklepath = dumpDir + "/" + filename + ".pkl"
	cPicklepath = dumpDir + "/" + filename + ".cpkl"
	#with open(picklepath, 'wb') as fid:
         #       pickle.dump(errDistMap, fid)

        #fid.close()
        with open(cPicklepath, 'wb') as fid:
                cPickle.dump(errDistMap, fid)
        fid.close()
	return (picklepath,cPicklepath)

def dumpSelectedFeaturesMap(colNameToIndexMap,dumpDir,tsvFName):
	#print "\n\n ------------------------- colNameToIndexMap", colNameToIndexMap
	onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
	filename = onlyName + "_selectedFeatures"
	picklepath = dumpDir + "/" + filename + ".pkl"
	cPicklepath = dumpDir + "/" + filename + ".cpkl"
        #with open(picklepath, 'wb') as fid:
         #       pickle.dump(colNameToIndexMap, fid)

        #fid.close()
        with open(cPicklepath, 'wb') as fid:
                cPickle.dump(colNameToIndexMap, fid)
        fid.close()
        return (picklepath,cPicklepath)

def loadSelectedFeaturesMap(dumpDir,isCPickle,tsvFName):
	onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
	baseDir = os.path.dirname(tsvFName)
        filename = onlyName + "_selectedFeatures"	
        pklFName = baseDir + "/" + filename
	colNameToIndexMap = None
	try:
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
	except IOError:
		return None

def dumpGoodTargetMap(goodTargetMap,dumpDir,tsvFName):
        #print "\n\n ------------------------- colNameToIndexMap", colNameToIndexMap
        onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
        filename = onlyName + "_goodTargets"
        picklepath = dumpDir + "/" + filename + ".pkl"
        cPicklepath = dumpDir + "/" + filename + ".cpkl"
        #with open(picklepath, 'wb') as fid:
        #        pickle.dump(goodTargetMap, fid)

        #fid.close()
        with open(cPicklepath, 'wb') as fid:
                cPickle.dump(goodTargetMap, fid)
        fid.close()
        return (picklepath,cPicklepath)

def loadGoodTargetMap(dumpDir,isCPickle,tsvFName):
        onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
        baseDir = os.path.dirname(tsvFName)
        filename = onlyName + "_goodTargets"
        pklFName = baseDir + "/" + filename
        goodTargetMap = None
        try:
                if(isCPickle):
                        pklFName = pklFName + ".cpkl"
                        with open(pklFName, 'rb') as fid:
                                goodTargetMap = cPickle.load(fid)
                        fid.close()
                        return goodTargetMap
                else:
                        pklFName = pklFName + ".pkl"
                        with open(pklFName, 'rb') as fid:
                                goodTargetMap = pickle.load(fid)
                        fid.close()
                        return goodTargetMap
        except IOError:
                return None

def dumpTargetErrMap(targetErrMap,dumpDir,tsvFName):
        #print "\n\n ------------------------- colNameToIndexMap", colNameToIndexMap
        onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
        filename = onlyName + "_targetErr"
        picklepath = dumpDir + "/" + filename + ".pkl"
        cPicklepath = dumpDir + "/" + filename + ".cpkl"
        #with open(picklepath, 'wb') as fid:
        #        pickle.dump(targetErrMap, fid)

        #fid.close()
        with open(cPicklepath, 'wb') as fid:
                cPickle.dump(targetErrMap, fid)
        fid.close()
        return (picklepath,cPicklepath)

def loadTargetErrMap(dumpDir,isCPickle,tsvFName):
        onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
        baseDir = os.path.dirname(tsvFName)
        filename = onlyName + "_targetErr"
        pklFName = baseDir + "/" + filename
        targetErrMap = None
        try:
                if(isCPickle):
                        pklFName = pklFName + ".cpkl"
                        with open(pklFName, 'rb') as fid:
                                targetErrMap = cPickle.load(fid)
                        fid.close()
                        return targetErrMap
                else:
                        pklFName = pklFName + ".pkl"
                        with open(pklFName, 'rb') as fid:
                                targetErrMap = pickle.load(fid)
                        fid.close()
                        return targetErrMap
        except IOError:
                return None

def loadErrorDistributionProfileMap(dumpDir, isCPickle,tsvFName):
	onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
	baseDir = os.path.dirname(tsvFName)
        filename = onlyName + "_errMapDump"	
        pklFName = baseDir + "/" + filename
	errProfileMap_loaded = None
	try:
		if(isCPickle):
			pklFName = pklFName + ".cpkl"
			print "Error prof file name:" + pklFName
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
	except IOError:
		return None

def dumpRegressorObjectDict(regressionObjectDict,dumpDir,tsvFName):
	onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
        filename = onlyName + "_regressionObjectDict"	
	picklepath = dumpDir + "/" + filename + ".pkl"
	cPicklepath = dumpDir + "/" + filename + ".cpkl"
        #with open(picklepath, 'wb') as fid:
        #        pickle.dump(regressionObjectDict, fid)

        #fid.close()
        with open(cPicklepath, 'wb') as fid:
                cPickle.dump(regressionObjectDict, fid)
        fid.close()
        return (picklepath,cPicklepath)	

def loadRegressorObjectDict(dumpDir, isCPickle,tsvFName):
	onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
	baseDir = os.path.dirname(tsvFName)
        filename = onlyName + "_regressionObjectDict"	
        pklFName = baseDir + "/" + filename
        regressionObjectDict_loaded = None
	try:
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
	except IOError:
		return None
