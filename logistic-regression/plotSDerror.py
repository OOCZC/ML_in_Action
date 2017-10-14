# -*- coding: utf-8 -*-
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logRegres

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    weightsHistory=zeros((500*m,n))
    for j in range(500):
        for i in range(m):
            h = logRegres.sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
            weightsHistory[j*m + i,:] = weights
    return weightsHistory

def stocGradAscent1(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.4
    weights = ones(n)   #initialize to all ones
    weightsHistory=zeros((40*m,n))
    for j in range(40):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = logRegres.sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            #print error
            weights = weights + alpha * error * dataMatrix[randIndex]
            weightsHistory[j*m + i,:] = weights
            del(dataIndex[randIndex])
    print weights
    return weightsHistory
    

dataMat,labelMat=logRegres.loadDataSet()
dataArr = array(dataMat)
myHist = stocGradAscent0(dataArr,labelMat)


n = shape(dataArr)[0] #number of points to create
xcord1 = []; ycord1 = []
xcord2 = []; ycord2 = []

markers =[]
colors =[]


fig = plt.figure() #创建图表
ax = fig.add_subplot(311)
#增加子图，三个数字的含义如下
#把画布分成3行1列，图像画在从左到右从上到下的第1块。
ax.plot(myHist[:,0])
plt.ylabel('X0')
ax = fig.add_subplot(312)
ax.plot(myHist[:,1])
plt.ylabel('X1')
ax = fig.add_subplot(313)
ax.plot(myHist[:,2])
plt.xlabel('iteration')
plt.ylabel('X2')
plt.show()