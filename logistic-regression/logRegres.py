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
	weights = ones((n,1)) #此处n为3
	#n行1列，都是1, array类型
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)
		error = (labelMat - h) #误差,越小越好
		weights = weights + alpha * dataMatrix.transpose() * error #  array*matrix 得到 matrix 类型 !!
		#更新回归系数
	return weights

	
def plotBestFit(weights):
    #weights本来为matrix类型，通过getA()变成了narray类型(3,1)
    #print type(weights[0]),'4444'
    #print weights
    import matplotlib.pyplot as plt
    #print type(weights),'0000000000000'
    dataMat,labelMat=loadDataSet()
    dataArr = numpy.array(dataMat)
    #print dataArr
    n = shape(dataArr)[0] 
    #shape()返回tuple类型 dataArr的长宽 这里去[0]即为多少个点
    #print type(n)
    #print n,'----------'
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()#创建图表1
    ax = fig.add_subplot(111) #增加子图，三个数字的含义如下
    #把画布分成1行1列，图像画在从左到右从上到下的第1块。
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    #描点

    #下面描线
    x = numpy.arange(-3.0, 3.0, 0.1) #等差数列
    #返回array数组，三个参数时，起点，终点，步长
    y = (-weights[0]-weights[1]*x)/weights[2]
    #这里weights[0]是array类型
    print weights[0],'37427429'
    ax.plot(x, y)

    plt.xlabel('X1'); plt.ylabel('X2'); #横纵坐标标签
    plt.show() #显示图像

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


if __name__ == '__main__':
	dataMat, labelMat = loadDataSet()
	weights = gradAscent(dataMat, labelMat)
	plotBestFit(weights.getA())  #matrix转化为narray
	#plotBestFit(weights)
