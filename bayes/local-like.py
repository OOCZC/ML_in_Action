#!/usr/bin/env python
# -*- coding: utf-8 -*-

import feedparser
import bayes
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
print 'ny download over'
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
#valueOfFeat = secondDict[key]
print 'sf download over'
#由于随机构建测试集，通过多次测试减小有误差
bayes.localWords(ny, sf)
bayes.localWords(ny, sf)
bayes.localWords(ny, sf)
bayes.localWords(ny, sf)
bayes.localWords(ny, sf)
bayes.localWords(ny, sf)
bayes.localWords(ny, sf)
