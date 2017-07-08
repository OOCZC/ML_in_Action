#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from math import exp
from numpy import mat, matrix
from numpy import shape,ones
import numpy
import types
def loadDataSet():
	dataMat = []; labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		#strip() 移除字符串头尾指定字符，默认空白字符
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		#这里设x0为1.0
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

def sigmoid(inX):
	return 1.0/(1+numpy.exp(-inX))

def gradAscent(dataMatIn, classLabels):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	#labelMat = mat(classLabels).T
	m,n = shape(dataMatrix)
	alpha = 0.001  #步长
	maxCycles = 500  #迭代次数
	weights = ones((n,1))
	#n行1列，都是1, array类型
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)
		error = (labelMat - h) #误差,越小越好
		weights = weights + alpha * dataMatrix.transpose() * error
		#更新回归系数
	return weights

if __name__ == '__main__':
	dataMat, labelMat = loadDataSet()
	print gradAscent(dataMat, labelMat)
