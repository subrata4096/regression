#!/usr/bin/python

import cPickle
# save the classifier
def dumpModel(tsvFName,target, model):
	import os.path
	locDir = os.path.dirname(tsvFName)
        onlyName = os.path.splitext(os.path.basename(tsvFName))[0]
        #print onlyName,  target
        outName  = onlyName + "_" + target + ".pkl"
        fullpath = os.path.join(locDir, outName)
        print fullpath
	with open(fullpath, 'wb') as fid:
    		cPickle.dump(model, fid)

        fid.close()    
	return fullpath
# load it again
def loadModel(pklFName):
	with open(pklFName, 'rb') as fid:
    		model_loaded = cPickle.load(fid)
	fid.close()
	return model_loaded


