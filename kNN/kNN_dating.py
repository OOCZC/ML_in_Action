#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt

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

def file2matrix():
	fr = open('datingTestSet.txt')
	arrayLines = fr.readlines()
	numberOfLines = len(arrayLines)
	returnMat = np.zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayLines:
		line = line.strip()
		#去除开头结尾空白字符
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		#returnMat[index,:]把三个数据放在index行，从0列开始放
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

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

def datingClassTest():
	hoRatio = 0.05
	datingDataMat,datingLabels = file2matrix()
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	#m是有多少行
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],\
	normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print 'the classifier came back with : %d, the\
	real answer is: %d' % (classifierResult, datingLabels[i])
		if (classifierResult != datingLabels[i]):
			errorCount += 1.0
	print 'the total error rate is: %f' % (errorCount/float(numTestVecs))
	
datingClassTest()
'''
datingDataMat,datingLabels = file2matrix()
normMat, ranges, minVals = autoNorm(datingDataMat)
print normMat
print ranges
print minVals

group, labels = createDataSet()
anw = classify0([2,3], group, labels, 3)
print anw

#绘图
datingDataMat,datingLabels = file2matrix()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1],\
15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
plt.show()
'''
