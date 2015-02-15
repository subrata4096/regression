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

def dumpErrorDistributionProfileMap(errDistMap):
	picklepath = "/home/mitra4/work/regression/errMapDump.pkl"
	cPicklepath = "/home/mitra4/work/regression/errMapDump.cpkl"
	with open(picklepath, 'wb') as fid:
                pickle.dump(errDistMap, fid)

        fid.close()
        with open(cPicklepath, 'wb') as fid:
                cPickle.dump(errDistMap, fid)
        fid.close()
	return (picklepath,cPicklepath)

def loadErrorDistributionProfileMap(pklFName, isCPickle):
	if(isCPickle):
		with open(pklFName, 'rb') as fid:
                	errProfileMap_loaded = cPickle.load(fid)
        	fid.close()
        	return errProfileMap_loaded
	else:
		with open(pklFName, 'rb') as fid:
                	errProfileMap_loaded = pickle.load(fid)
        	fid.close()
        	return errProfileMap_loaded
		
