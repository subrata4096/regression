#!/usr/bin/python
import os
from abc import ABCMeta, abstractmethod

def setActiveDumpDirectory(dataFile):
	fileWithoutExt = os.path.splitext(dataFile)[0]
        workingDir = os.path.join(os.path.expanduser("~"),"errorFigDump")
        activeDumpDir = os.path.join(workingDir,os.path.basename(fileWithoutExt))
        print activeDumpDir
	return activeDumpDir



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

def initializeGlobalObjects():
	globalObjectsContainerClass.globalObjectMap["activeDumpDirectory"] = ''
	globalObjectsContainerClass.globalObjectMap["inputColumnNames"] = []
	globalObjectsContainerClass.globalObjectMap["measuredColumnNames"] = []
	globalObjectsContainerClass.globalObjectMap["outputColumnNames"] = []
	globalObjectsContainerClass.globalObjectMap["regressionDict"] = {}
	globalObjectsContainerClass.globalObjectMap["inputColumnNameToIndexMapFromFile"] = {}
	globalObjectsContainerClass.globalObjectMap["measuredColumnNameToIndexMapFromFile"] = {}
	globalObjectsContainerClass.globalObjectMap["outputColumnNameToIndexMapFromFile"] = {}

	globalObjectsContainerClass.globalObjectMap["inputIndexToFieldNameMap"] = {}
	globalObjectsContainerClass.globalObjectMap["measuredIndexToFieldNameMap"] = {}
	globalObjectsContainerClass.globalObjectMap["outputIndexToFieldNameMap"] = {}


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
inputColumnNames = ['in1', 'in2', 'in3']
#measuredColumnNames = ['module:measure:PAPI:PAPI_BR_CN','module:measure:PAPI:PAPI_FP_OPS','module:measure:PAPI:PAPI_TOT_INS','module:measure:time:time']
#measuredColumnNames = ['module:measure:time:time','module:measure:PAPI:PAPI_BR_CN','module:measure:PAPI:PAPI_FP_OPS','module:measure:PAPI:PAPI_TOT_INS']
#measuredColumnNames = ['module:measure:time:time','module:measure:PAPI:PAPI_TOT_INS']
#measuredColumnNames = ['module:measure:PAPI:PAPI_L2_TC_MR','module:measure:PAPI:PAPI_TOT_INS','module:measure:RAPL:Elapsed']
#measuredColumnNames = ['module:measure:PAPI:PAPI_L2_TC_MR','module:measure:PAPI:PAPI_TOT_INS','module:measure:RAPL:Elapsed']
#measuredColumnNames = ['module:measure:PAPI:PAPI_L2_TC_MR','module:measure:PAPI:PAPI_TOT_INS','module:measure:RAPL:EDP_S0','module:measure:RAPL:EDP_S1','module:measure:RAPL:Elapsed','module:measure:RAPL:Energy_CPU_S0','module:measure:RAPL:Energy_CPU_S1','module:measure:RAPL:Energy_DRAM_S0','module:measure:RAPL:Energy_DRAM_S1','module:measure:RAPL:Power_CPU_S0','module:measure:RAPL:Power_CPU_S1','module:measure:RAPL:Power_DRAM_S0','module:measure:RAPL:Power_DRAM_S1']
#measuredColumnNames = ['module:measure:PAPI:PAPI_TOT_INS','module:measure:time:time']
#measuredColumnNames = ['module:measure:time:time','module:input:1:u_cut']
#measuredColumnNames = ['module:measure:RAPL:Elapsed','module:measure:RAPL:EDP_S0']
measuredColumnNames = ['m1','m2']
#outputColumnNames = ['module:output:0:TotalAbsDiff','module:output:1:numCycles']
#outputColumnNames = ['module:output:0:TotalAbsDiff','module:output:1:numCycles']
#outputColumnNames = ['module:output:0:MaxRelDiff','module:output:0:TotalAbsDiff']
#outputColumnNames = ['module:measure:PAPI:PAPI_BR_CN','module:measure:PAPI:PAPI_FP_OPS']
#outputColumnNames = ['module:output:0:FOM','module:output:0:MaxAbsDiff','module:output:0:MaxRelDiff','module:output:0:TotalAbsDiff','module:output:1:numCycles']
outputColumnNames = ['o1','o2','o3']
#outputColumnNames = ['o1','o2']

def initializeGlobalObjects():
        globalObjectsContainerClass.globalObjectMap["activeDumpDirectory"] = ''
        globalObjectsContainerClass.globalObjectMap["inputColumnNames"] = []
        globalObjectsContainerClass.globalObjectMap["measuredColumnNames"] = []
        globalObjectsContainerClass.globalObjectMap["outputColumnNames"] = []
        globalObjectsContainerClass.globalObjectMap["regressionDict"] = {}
        globalObjectsContainerClass.globalObjectMap["inputColumnNameToIndexMapFromFile"] = {}
        globalObjectsContainerClass.globalObjectMap["measuredColumnNameToIndexMapFromFile"] = {}
        globalObjectsContainerClass.globalObjectMap["outputColumnNameToIndexMapFromFile"] = {}

        globalObjectsContainerClass.globalObjectMap["inputIndexToFieldNameMap"] = {}
        globalObjectsContainerClass.globalObjectMap["measuredIndexToFieldNameMap"] = {}
        globalObjectsContainerClass.globalObjectMap["outputIndexToFieldNameMap"] = {}

	#this is a global container for error datastructure
	#targetName vs TargetErrorData map
	globalObjectsContainerClass.globalObjectMap["TargetErrorDataMap"] = {}
	 #This is a map where initial samples for test and train are kept.
        #Later it is populated with prediction functions and prediction errors.
        #The top level key is "target-name". The content DataStructure "TargetErrorData"

        #This map keeps the calculated error profile for each target for each profile along with curve-fitted error function
	#top level key is target name. 2nd level key is feature name. Then the content is "errorDistributionProfile"
	globalObjectsContainerClass.globalObjectMap["ErrorDistributionProfileMapForTargetAndFeature"] = {}
	
	setGlobalObject("inputColumnNames",inputColumnNames)
	setGlobalObject("measuredColumnNames",measuredColumnNames)
	setGlobalObject("outputColumnNames",outputColumnNames)
