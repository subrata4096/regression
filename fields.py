#!/usr/bin/python
import os
from abc import ABCMeta, abstractmethod
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def makeDirectoriesRecursively(dirName):
	try:
		os.makedirs(dirName)
	except:
		pass

def setActiveDumpDirectory(dataFile):
	fileWithoutExt = os.path.splitext(dataFile)[0]
        workingDir = os.path.join(os.path.expanduser("~"),"errorFigDump")
        activeDumpDir = os.path.join(workingDir,os.path.basename(fileWithoutExt))
        print activeDumpDir
	return activeDumpDir

def getSelectedColumnNames(selectedOrigColIndexMap):
        selectedColumnNameMap = {}
        for colIdx in selectedOrigColIndexMap.keys():
                if(selectedOrigColIndexMap[colIdx] == False):
                        continue

                colName = getGlobalObject("inputColumnIndexToNameMapFromFile")[colIdx]
                featureIndex = getGlobalObject("columnIndexToInArrIndexMap")[colIdx]
                selectedColumnNameMap[colName] = featureIndex
        #end for
        #return getSortedTupleFromDictionary(selectedColumnNameMap)
        return selectedColumnNameMap


def getInputParameterNameFromFeatureIndex(featureIndex):
        colIdx = getColumnIndexFromFeatureIndex(featureIndex)
        #print "feature idx = ", featureIndex, " colIdx = " , colIdx
        return getInputParameterNameFromColumnIndex(colIdx)

def getInputParameterNameFromColumnIndex(columnIndex):
        #NOTE! This function might have bugs. This index may not be the actual index we are looking for
        #paramName = InputColumnIndexToNameMap[columnIndex]
        #paramName = inputColumnNames[columnIndex]
        #^^ was old and inaccurate
        #get the index from updated index to name map
        #print getGlobalObject("inputIndexToFieldNameMap")
        #paramName = getGlobalObject("inputIndexToFieldNameMap")[columnIndex]
        #print getGlobalObject("inputColumnIndexToNameMapFromFile")
        paramName = getGlobalObject("inputColumnIndexToNameMapFromFile")[columnIndex]
        return paramName

def getColumnIndexFromFeatureIndex(featureIndex):
	if(len(getGlobalObject("InArrIndexToColumnIndexMap")) > 0):
        	return getGlobalObject("InArrIndexToColumnIndexMap")[featureIndex]
	else:
		print "here"
		print getGlobalObject("columnIndexToInArrIndexMap")
        	colIdxToInArrIdxMap = getGlobalObject("columnIndexToInArrIndexMap")
        #print colIdxToInArrIdxMap
        	for colIdx,inArrIdx in colIdxToInArrIdxMap.iteritems():
        		if(inArrIdx == featureIndex):
              	          return colIdx
        return -1

class globalObjectsContainerClass:
	__metaclass__ = ABCMeta
	globalObjectMap = {}
	@abstractmethod
	def name(self):
		pass

#global activeDumpDirectory

#global inputColumnNames
#global measuredColumnNames
#global outputColumnNames
#global regressionDict

#global inputColumnNameToIndexMapFromFile
#global measuredColumnNameToIndexMapFromFile
#global outputColumnNameToIndexMapFromFile
#inputColumnNameToIndexMapFromFile = {}
#global inputIndexToFieldNameMap
#global measuredIndexToFieldNameMap
#global outputIndexToFieldNameMap

def setGlobalObject(objectName,objectRef):
	globalObjectsContainerClass.globalObjectMap[objectName] = objectRef

def getGlobalObject(objectName):
	return globalObjectsContainerClass.globalObjectMap[objectName]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_input(s):
        pos = s.find("input:")
        if(pos > -1):
		if(s.find("input::eKinetic") != -1):
			print "We will not consider input::eKinetic as part of inputs. Skipping..\n"
                	return False
		elif(s.find("input::ePotential") != -1):
			print "We will not consider input::ePotential as part of inputs. Skipping..\n"
                	return False
		elif(s.find("input::momStdDev") != -1):
			print "We will not consider input::momStdDev as part of inputs. Skipping..\n"
                	return False
		elif(s.find("input::posStdDev") != -1):
			print "We will not consider input::posStdDev as part of inputs. Skipping..\n"
                	return False
		else:
                	return True

        else:
                return False
def is_measure(s):
        pos = s.find("measure:")
        if(pos > -1):
		if(s.find("measure:timestamp") != -1):
			print "We will not consider measure:timestamp as part of target. Skipping..\n"
                	return False
		else:
			return True
        else:
                return False
def is_output(s):
        pos = s.find("output:")
        if(pos > -1):
                return True
        else:
                return False

def parseFields(fname):
	print "Automatically identify input, measurement, output fields from naming convention"
        inputFields = []
        measureFields = []
        outputFields = []
        with open(fname) as f:
                firstLine = f.readline()
                secondLine = f.readline()
                headerline = firstLine.strip()
                valueline = secondLine.strip()
		#print headerline
                headerFields = headerline.split("\t")
                valueFields = valueline.split("\t")
                numOfFields = len(headerFields)
                for idx in range(numOfFields):
                        header = headerFields[idx]
                        value = valueFields[idx]
			#print header, idx
                        if(False == is_number(value)):
                                continue
                        elif(is_input(header)):
                                inputFields.append(header)
                        elif(is_measure(header)):
                                measureFields.append(header)
                        elif(is_output(header)):
                                outputFields.append(header)
                        else:
                                print "Something is wrong with field: " + header
                                #exit(0)

	print "\n****** For file: " + fname
        print "--> Input fields: " + str(inputFields)
        print "--> Measure fields: " + str(measureFields)
        print "--> Output fields: " + str(outputFields)

        return inputFields,measureFields,outputFields


#inputColumnNames = ['module:input:0:length','module:pub_input::balance','module:pub_input::cost','module:pub_input::dtfixed','module:pub_input::dtmax','module:pub_input::iter','module:pub_input::its','module:pub_input::numElem','module:pub_input::numNode','module:pub_input::numReg','module:pub_input::nx','module:pub_input::phase','module:pub_input::u_cut']
#inputColumnNames = ['module:pub_input::iter','module:pub_input::numElem','module:pub_input::numNode','module:pub_input::nx','module:pub_input::phase']
#inputColumnNames = ['module:pub_input::balance','module:pub_input::cost','module:pub_input::dtfixed','module:pub_input::dtmax','module:pub_input::iter','module:pub_input::its','module:pub_input::numElem','module:pub_input::numNode','module:pub_input::numReg','module:pub_input::nx','module:pub_input::phase','module:pub_input::u_cut']
#inputColumnNames = ['module:input:0:numRanks','module:input:0:nx']
#inputColumnNames = ['module:input:0:numRanks','module:input:0:nx']
#inputColumnNames = ['module:input:0:nx','module:input:0:real_prec']
#inputColumnNames = ['module:input:0:balance','module:input:0:cost','module:input:0:dtfixed','module:input:0:dtmax','module:input:0:its','module:input:0:numNodes','module:input:0:numRanks','module:input:0:numReg','module:input:0:numZones','module:input:0:nx','module:input:0:powercap','module:input:0:rank','module:input:0:real_prec']
#inputColumnNames = ['module:input:0:balance','module:input:0:cost','module:input:0:dtfixed','module:input:0:dtmax','module:input:0:host','module:input:0:its','module:input:0:numNodes','module:input:0:numRanks', 'module:input:0:numReg','module:input:0:numZones', 'module:input:0:nx','module:input:0:powercap','module:input:0:rank','module:input:0:real_prec','module:input:0:system','module:input:1:MaxAbsDiff','module:input:1:MaxRelDiff','module:input:1:TotalAbsDiff','module:input:1:iter','module:input:1:numElem','module:input:1:numNode','module:input:1:phase','module:input:1:u_cut']
#inputColumnNames = ['module:pub_input::iStep','module:pub_input::lat']
#inputColumnNames = ['module:input:0:ii','module:pub_input::dt','module:pub_input::eKinetic','module:pub_input::ePotential','module:pub_input::iStep','module:pub_input::lat','module:pub_input::momStdDev','module:pub_input::posStdDev']
#inputColumnNames = ['module:input:0:ii','module:pub_input::dt','module:pub_input::iStep','module:pub_input::lat']
#inputColumnNames = ['module:pub_input::dt','module:pub_input::lat','module:input:0:iStep']
#inputColumnNames = ['module:pub_input::dt','module:pub_input::lat']
#inputColumnNames = ['module:input:0:dt','module:input:0:lat']
#inputColumnNames = ['in1', 'in2', 'in3']
#inputColumnNames = ['in1', 'in2']
#measuredColumnNames = ['module:measure:PAPI:PAPI_BR_CN','module:measure:PAPI:PAPI_FP_OPS','module:measure:PAPI:PAPI_TOT_INS','module:measure:time:time']
#measuredColumnNames = ['module:measure:time:time','module:measure:PAPI:PAPI_BR_CN','module:measure:PAPI:PAPI_FP_OPS','module:measure:PAPI:PAPI_TOT_INS']
#measuredColumnNames = ['module:measure:time:time','module:measure:PAPI:PAPI_TOT_INS']
#measuredColumnNames = ['module:measure:PAPI:PAPI_L2_TC_MR','module:measure:PAPI:PAPI_TOT_INS','module:measure:RAPL:Elapsed']
#measuredColumnNames = ['module:measure:PAPI:PAPI_L2_TC_MR','module:measure:PAPI:PAPI_TOT_INS','module:measure:RAPL:Elapsed']
#measuredColumnNames = ['module:measure:PAPI:PAPI_L2_TC_MR','module:measure:PAPI:PAPI_TOT_INS','module:measure:RAPL:EDP_S0','module:measure:RAPL:EDP_S1','module:measure:RAPL:Elapsed','module:measure:RAPL:Energy_CPU_S0','module:measure:RAPL:Energy_CPU_S1','module:measure:RAPL:Energy_DRAM_S0','module:measure:RAPL:Energy_DRAM_S1','module:measure:RAPL:Power_CPU_S0','module:measure:RAPL:Power_CPU_S1','module:measure:RAPL:Power_DRAM_S0','module:measure:RAPL:Power_DRAM_S1']
#measuredColumnNames = ['module:measure:PAPI:PAPI_TOT_INS','module:measure:time:time']
#measuredColumnNames = ['module:measure:time:time','module:input:1:u_cut']
#measuredColumnNames = ['module:measure:RAPL:Elapsed','module:measure:RAPL:EDP_S0']
#measuredColumnNames = ['m1','m2']
#outputColumnNames = ['module:output:0:TotalAbsDiff','module:output:1:numCycles']
#outputColumnNames = ['module:output:0:TotalAbsDiff','module:output:1:numCycles']
#outputColumnNames = ['module:output:0:MaxRelDiff','module:output:0:TotalAbsDiff']
#outputColumnNames = ['module:measure:PAPI:PAPI_BR_CN','module:measure:PAPI:PAPI_FP_OPS']
#outputColumnNames = ['module:output:0:FOM','module:output:0:MaxAbsDiff','module:output:0:MaxRelDiff','module:output:0:TotalAbsDiff','module:output:1:numCycles']
#outputColumnNames = ['o1','o2','o3']
#outputColumnNames = ['o1','o2']

def initializeGlobalObjects(dataFileName):
	
        globalObjectsContainerClass.globalObjectMap["baseModuleName"] = ''
        globalObjectsContainerClass.globalObjectMap["activeDumpDirectory"] = ''
        
	globalObjectsContainerClass.globalObjectMap["inputColumnNames"] = []
        globalObjectsContainerClass.globalObjectMap["measuredColumnNames"] = []
        globalObjectsContainerClass.globalObjectMap["outputColumnNames"] = []
        
	globalObjectsContainerClass.globalObjectMap["regressionDict"] = {}
        
	globalObjectsContainerClass.globalObjectMap["columnIndexToInArrIndexMap"] = {}
	globalObjectsContainerClass.globalObjectMap["InArrIndexToColumnIndexMap"] = {}
	globalObjectsContainerClass.globalObjectMap["columnIndexToMsrArrIndexMap"] = {}
	globalObjectsContainerClass.globalObjectMap["columnIndexToOutArrIndexMap"] = {}

        globalObjectsContainerClass.globalObjectMap["inputColumnNameToIndexMapFromFile"] = {}
        globalObjectsContainerClass.globalObjectMap["inputColumnIndexToNameMapFromFile"] = {}

        globalObjectsContainerClass.globalObjectMap["measuredColumnNameToIndexMapFromFile"] = {}
        globalObjectsContainerClass.globalObjectMap["measuredColumnIndexToNameMapFromFile"] = {}

        globalObjectsContainerClass.globalObjectMap["outputColumnNameToIndexMapFromFile"] = {}
        globalObjectsContainerClass.globalObjectMap["outputColumnIndexToNameMapFromFile"] = {}

        globalObjectsContainerClass.globalObjectMap["inputIndexToFieldNameMap"] = {}
        globalObjectsContainerClass.globalObjectMap["measuredIndexToFieldNameMap"] = {}
        globalObjectsContainerClass.globalObjectMap["outputIndexToFieldNameMap"] = {}
        
	globalObjectsContainerClass.globalObjectMap["selectedOriginalColIndexMap"] = {}

	#this is a global container for error datastructure
	#targetName vs TargetErrorData map
	globalObjectsContainerClass.globalObjectMap["TargetErrorDataMap"] = {}
	 #This is a map where initial samples for test and train are kept.
        #Later it is populated with prediction functions and prediction errors.
        #The top level key is "target-name". The content DataStructure "TargetErrorData"

        #This map keeps the calculated error profile for each target for each profile along with curve-fitted error function
	#top level key is target name. 2nd level key is feature name. Then the content is "errorDistributionProfile"
	globalObjectsContainerClass.globalObjectMap["ErrorDistributionProfileMapForTargetAndFeature"] = {}
	
	globalObjectsContainerClass.globalObjectMap["goodTargetMap"] = {}
	
	#This routine automatically infers input,measurement,output columns from there names, based on the Sight naming convetion
	
	inputColumnNames,measuredColumnNames,outputColumnNames = parseFields(dataFileName)
	
	#ELSE: use the above values when commented out, for e.g. during test
	
	setGlobalObject("inputColumnNames",inputColumnNames)
	setGlobalObject("measuredColumnNames",measuredColumnNames)
	setGlobalObject("outputColumnNames",outputColumnNames)
