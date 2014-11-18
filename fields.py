#!/usr/bin/python


global inputColumnNames
global measuredColumnNames
global outputColumnNames
global regressionDict
regressionDict = {}

inputColumnNames = []
measuredColumnNames = []
outputColumnNames = []

inputColumnNames = ['module:input:0:length','module:pub_input::balance','module:pub_input::cost','module:pub_input::dtfixed','module:pub_input::dtmax','module:pub_input::iter','module:pub_input::its','module:pub_input::numElem','module:pub_input::numNode','module:pub_input::numReg','module:pub_input::nx','module:pub_input::phase','module:pub_input::u_cut']
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
#measuredColumnNames = ['module:measure:time:time','module:measure:PAPI:PAPI_BR_CN','module:measure:PAPI:PAPI_FP_OPS','module:measure:PAPI:PAPI_TOT_INS']
measuredColumnNames = ['module:measure:time:time','module:measure:PAPI:PAPI_TOT_INS']
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
outputColumnNames = ['module:measure:PAPI:PAPI_BR_CN','module:measure:PAPI:PAPI_FP_OPS']
#outputColumnNames = ['module:output:0:FOM','module:output:0:MaxAbsDiff','module:output:0:MaxRelDiff','module:output:0:TotalAbsDiff','module:output:1:numCycles']
#outputColumnNames = ['o1','o2','o3']
