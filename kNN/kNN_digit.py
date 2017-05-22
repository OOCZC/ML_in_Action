#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	#axis=1是横着求和
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount={}
	for i in range(k):
		voteLabel = labels[sortedDistIndicies[i]]
		classCount[voteLabel] = classCount.get(voteLabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(),\
 key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def img2vector(filename):
	returnVect = np.zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def autoNorm(dataSet):
	#归一化数值
	#dataSet为n*3矩阵
	minVals = dataSet.min(0)
	#print minVals
	#从列中选取最小值，而不是当前行最小值，行最小min(1)
	maxVals = dataSet.max(0)
	#print maxVals
	ranges = maxVals - minVals
	#print ranges
	normDataSet = np.zeros(np.shape(dataSet))
	#shape(dataSet)返回矩阵的形状,eg: (3,4)
	m = dataSet.shape[0] #dataSet第一维的长度
	#print m
	normDataSet = dataSet - np.tile(minVals, (m,1))
	normDataSet = normDataSet/np.tile(ranges, (m,1))
	#print normDataSet 
	return normDataSet, ranges, minVals

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	#listdir返回文件夹所有文件名的list
	m = len(trainingFileList)
	trainingMat = np.zeros((m, 1024))
	#一行存储一个图像
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s'\
	% fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0] #take off .txt
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		#print "the classifier came back with: %d, the real answer is: %d"\
		 #% (classifierResult, classNumStr)
		if (classifierResult != classNumStr): 
			errorCount += 1.0
	print "\nthe total number of errors is: %d" % errorCount
	print "\nthe total error rate is: %f" % (errorCount/float(mTest))

handwritingClassTest()
#testVector = img2vector('trainingDigits/0_13.txt')
#print testVector[0,0:1000]
#print testVector[0:1000]不能全部输出


