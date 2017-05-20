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

def createDataSet():
	group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def file2matrix():
	fr = open('datingTestSet.txt')
	arrayLines = fr.readlines()
	numberOfLines = len(arrayLines)
	returnMat = np.zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayLines:
		line = line.strip()
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
	print minVals
	#从列中选取最小值，而不是当前行最小值，行最小min(1)
	maxVals = dataSet.max(0)
	print maxVals
	ranges = maxVals - minVals
	print ranges
	normDataSet = np.zeros(np.shape(dataSet))
	#shape(dataSet)返回矩阵的形状,eg: (3,4)
	m = dataSet.shape[0] #dataSet第一维的长度
	print m
	normDataSet = dataSet - np.tile(minVals, (m,1))
	normDataSet = normDataSet/np.tile(ranges, (m,1))
	print normDataSet 
	return normDataSet, ranges, minVals

datingDataMat,datingLabels = file2matrix()
autoNorm(datingDataMat)
'''
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
