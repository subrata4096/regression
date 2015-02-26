#!/usr/bin/python
import scipy

def getAreaUnderCurveForNormalDistribution(zValue):
	area = 1.0 - (scipy.stats.norm.sf(zValue)*2)
	return area
