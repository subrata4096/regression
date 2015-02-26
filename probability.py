#!/usr/bin/python
import scipy

def getAreaUnderCurveForNormalDistribution(zValue):
	area = 1.0 - (scipy.stats.norm.sf(zValue)*2)
	return area

#this should calculate erf(n/sqrt(2)) : Ref wikipedia http://en.wikipedia.org/wiki/Normal_distribution

erfFunctionMap = {}
erfFunctionMap[0.0] = 0.0
erfFunctionMap[0.5] = 0.383
erfFunctionMap[1.0] = 0.682
erfFunctionMap[1.5] = 0.866
erfFunctionMap[2.0] = 0.954
erfFunctionMap[2.5] = 0.987
erfFunctionMap[3.0] = 0.997

def getProbabilityOfNormalDeviate(stdDev_multiplier)
	n = abs(stdDev_multiplier)

	if n in erfFunctionMap.keys():
		return erfFunctionMap[n]

	if((0.0 < n) && (n < 0.5)):
		return (erfFunctionMap[0.0] + erfFunctionMap[0.5])/2
	elif((0.5 < n) && (n < 1.0)):
		return (erfFunctionMap[0.5] + erfFunctionMap[1.0])/2
	elif((1.0 < n) && (n < 1.5)):
		return (erfFunctionMap[1.0] + erfFunctionMap[1.5])/2
	elif((1.5 < n) && (n < 2.0)):
		return (erfFunctionMap[1.5] + erfFunctionMap[2.0])/2
	elif((2.0 < n) && (n < 2.5)):
		return (erfFunctionMap[2.0] + erfFunctionMap[2.5])/2
	elif((2.5 < n) && (n < 3.0)):
		return (erfFunctionMap[2.5] + erfFunctionMap[3.0])/2
	else:
		return 1
