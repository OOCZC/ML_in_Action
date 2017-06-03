#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from math import log
'''
计算香农熵，只计算标签。
dataSet是二维List
即[[1,0,1],[0,1,1],[0,0,1]]
'''
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt

'''
dataSet的最后一个是后面需要判断的属性
'''
def createDataSet():
	dataSet = [[1, 1, 'yes'],
		[1, 1, 'yes'],
		[1, 0, 'no'],
		[0, 1, 'no'],
		[0, 1, 'no']]
	#通过前两个属性，判断最后一个。
	labels = ['no surfacing', 'flippers']	
	#前两个属性是什么，后面不太需要用到
	return dataSet, labels
'''
把axis轴值为value的数据分割出来，axis轴的数据不保留
'''
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			#排除axis轴的数据
			retDataSet.append(reducedFeatVec)
	return retDataSet

'''
选择最好的划分数据集的特征
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      
    #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    #原始香农熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        
    #iterate over all the features
        featList = [example[i] for example in dataSet]
        #dataSet中所有第i个特征
        uniqueVals = set(featList)  
        #get a set of unique values
        newEntropy = 0.0
        #entropy == 熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy 
        #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):    
        #compare this to the best gain so far
            bestInfoGain = infoGain     
        #if better than current best, set to best
            bestFeature = i
    return bestFeature                
    #returns an integer

'''
classList 是一维List
返回最多出现的标签。
即多数表决。
'''

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
创建决策树。
两个参数与createDataSet返回的参数相同
'''

def createTree(dataSet,labels):
    #函数参数是列表类型时，按引用方式传递，即改变原值。
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: 
        #stop splitting when there are no more features in dataSet
        #这里只剩最后的判断属性了。没有特征了
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #bestFeatLabel是标签的名字 
    myTree = {bestFeatLabel:{}}
    #字典存储无序键值对
    #myTree[bestFeatLabel]={}
    #等价与上面的语句，都是向字典中添加键值对。这里的值为空字典
    del(labels[bestFeat])
    #删除list中的这个元素
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:] 
        #copy all of labels, so trees don't mess up(弄乱) existing labels
        #如果直接给下面的函数参数labels，则每次经过myTree的labels都将不同。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
        #这里myTree[bestFeatLabel]对应的仍为一个字典
    return myTree                            


dataSet, labels = createDataSet()
trees = createTree(dataSet, labels)
print trees


'''
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
	


myDat, label = createDataSet()
print calcShannonEnt(myDat)
'''
