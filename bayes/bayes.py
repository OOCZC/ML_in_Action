#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
import uniout  #让list正常输出中文，不显示unicode

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]
	
'''
dataSet是loadDataSet()中的postingList
'''
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def bagOfWords2Vec(vocabList, inputSet):
    #词袋模型
    #在区域倾向测试中使用词袋模型正确率0.9,setOfWordsVec正确率为0.45
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

'''

trainNB(trainMatrix, trainCategory):
训练出模型

#trainMatrix是二维array类型，为词向量矩阵,每一行为词向量
array((1,0,1,0,1,0),(0,1,1,1,0,1),(0,0,1,1,1,0))
#trainCategory是一维array类型，标记每个词向量的类型
array(1,0,0)

p0Vec,p1Vec:array型一维数组，与词向量等长，表示在0/1型短
信中各个词出现的概率，即P(w|c)
pAbusive:浮点数，表示垃圾短信的概率，即P(c1)

'''
def trainNB0(trainMatrix, trainCategory):
	# trainMatrix是二维array类型，为词向量矩阵
	# trainCategory是一维array类型
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0]) #总词数
	pAbusive = sum(trainCategory) / float(numTrainDocs) 
	#spam短信的概率
	p0Num = ones(numWords) #array数组，初始化为1
	p1Num = ones(numWords) #防止多个概率相乘为0
	p0Denom = 2.0 #浮点型数据
	p1Denom = 2.0 #垃圾短信总词数
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vec = log(p1Num/p1Denom) #array数组除浮点数，得array数组
	p0Vec = log(p0Num/p0Denom) #取log，之后相加即可
	#每个词在普通短信中出现的概率，p(w|c0)
	return p0Vec,p1Vec,pAbusive
	# 两个向量，一个垃圾短信概率


'''

classifyNB(vec2Classify, p0Vec, p1Vec, pClass):
判断vec2Classify这个词向量所属类型

#vec2Classify是词向量，(1,0,0,1,0,1)
#p0Vec,p1Vec,PClass 为trainNB的三个输出

'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	#判断vec2Classify向量的分类
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	# sum(vec2Classify * p1Vec)为log(p(w|c))
	# p1为log( p(w|c)*p(c1) )
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 1 #为垃圾短信
	else:
		return 0 #为正常短信


def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

    
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    #以下为交叉验证
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText


'''
从个人广告中获取区域倾向
'''

'''
vocabList是去重后的fullText。
返回出现最多的前30个单词的键值对
eg. apple:3 
'''
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    #feed1, feed0是dict类型，由feedparser.parse('http://xx.com/in.rss')返回得来
    #import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    #vocabList = set(fullText) 也可吧?
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    #vocabList是去重后的fullText
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]   #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        #随机构建测试集及训练集
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V
    #这里的vocabList是去除了前30单词的。

'''
nf = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
getTopWords(nf,sf,10)
这里10为输出概率最高的前10个
'''
def getTopWords(ny,sf,num):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF"
    for i in range(10):
        print sortedSF[i][0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY"
    for i in range(10):
        print sortedSF[i][0]
