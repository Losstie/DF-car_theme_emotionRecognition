#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@project: DF_Car
@file:car_model_09-09.py.py
@author: dujiahao
@create_time: 2018/9/9 7:59
@description:
"""
import numpy as np
import pandas as pd
from  gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import jieba
import logging
import os
import re


class Word2vec_model(object):

    def __init__(self):
        # 导入停用词
        stpwrd_dic = open("..\data\stop_words.txt", 'r')
        stpwrd_content = stpwrd_dic.read()
        # 将停用词表转换为list
        stpwrdlst = stpwrd_content.splitlines()
        stpwrd_dic.close()
        suggest = open("..\data\/newdict.txt",'r',encoding="UTF-8")
        suggest_freq = suggest.read()
        suggest_freq = suggest_freq.splitlines()
        suggest.close()
        # 添加词典
        for sugg in suggest_freq:
            jieba.suggest_freq(sugg, True)


        self.stopwords = stpwrdlst
        self.suggest_freq = suggest_freq
        self.keywords_subject = [u"动力",u"油耗",u"价格",u"内饰",u"配置",u"安全性",u"外观",u"操控",
                                 u"空间",u"舒适性"]
        self.subject_eng = ["price","comfort","configuration","exterior","fuelConsumption",
                            "interior","manpulation","power","safety","space"]
        self.subjectRelated = {
            "price":["价格","价钱","价位","有钱","土豪","优惠","钱","现金","性价比","便宜","贵","高昂","降价",
                     "涨价","定价","报价","物美价廉","几千","几万","几十万","几百万","几千万","贵","值","礼包",
                     "预算","省","万","花销"],
            "comfort":["舒适","舒适性","舒服","安静","凉","声音","噪音","效果","软","硬","忍受","异响","结实","静音","胎噪",
                       "噪","风噪","响","问题","冷","体验","漏水","抖","清理","老化","影响","空调","受不了","味",
                       "太差","一般","座椅"],
            "configuration":["配置","导航","中控","CD","显示屏幕","显示屏","音响","黑屏","收音机","倒车影像","雷达",
                             "方向盘","玻璃","座椅","启停","后窗除雾","四驱","手挡","后备箱","行程安全系统","自动","手动",
                             "保险杠","后尾翼","车载地图","发动机","LD","一键启动","电动尾门","顶配","低配","动力","喇叭",
                             "性能","抖","主机","机油滤芯","刹车片","轴承","高配","摆臂","出风口","手刹","新款","变速箱"],
            "exterior":["外观","难看","颜色","车漆","银色","白色","黑色","红色","寒冰银","好看","蓝色","车身","光泽度",
                        "银色","时尚","前脸","绿色","内饰","不错","丑","漂亮","颜值","空间","外形","蔚蓝","蜡","天窗","后视镜",
                        "遮阳档","车身","远观图","近观图","白","蓝","黑","红","银","亮点","棕色","獠牙","豪华版","低调"],
            "fuelConsumption": ["节油","油耗","省油","费油","油","排量","汽油","烧","耗"],
            "interior": ["内饰","原装","改装","座椅","做工","细节","驾驶座","材料","豪华","脱皮","后排","前排","坐垫"],

            "manpulation": ["轮胎","操控","提速","减速","动平衡","底盘","操控性","性能","四驱","专业","轮毂","漂移","甩尾",
                            "油门","换挡","刹车","手刹","涡轮","动力","通过性","方向盘","变速箱"],
            "power": ["动力","灵敏","延迟","熄火","油门","机油","公里","发动机","马力","秒","超车","高速","暴力驾驶","爬坡",
                      "坡","排量","变速"],
            "safety": ["安全","冰路","抱死","刹车","刹车油","刹车片","气囊","抓地","刹车盘","追尾"],
            "space":["空间","放","后备箱"]
        }
        self.keywords_emotion = [-1,0,1]
        self.positive_words = [u"杠杠的",u"非常好",u"好",u"给力"]
        self.negative_words = [u"坑",u"不舒服"]

    # 数据预处理
    def pre_process(self,data):
        pass
    # 分词及去停用词
    def segSentence(self, s):
        words = list(jieba.cut(s))
        words = list(filter(lambda w: w not in self.stopwords, words))
        words = ' '.join(words)
        words = ' '.join(words.split())  # remove blank
        return words
    # 训练word2vec模型
    def word2vec_train(self,dataPath):
        with open(dataPath,'r',encoding='UTF-8') as f:
            traindata = f.read()
            document_cut = jieba.cut( traindata)
            result = ' '.join(document_cut)
            with open('..\data\extractData\documentcut.txt', 'w',encoding="UTF-8") as f2:
                f2.write(result)
        f.close()
        f2.close()
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = LineSentence('..\data\extractData\documentcut.txt')
        model = Word2Vec(sentences, min_count=1, seed=0, size=100,workers=1,iter=20)
        model.save("..\data\model_1")



    # 获得单个句子的词向量表示
    def get_vector(self, words):
        # 加载词向量模型
        model = Word2Vec.load("..\data\model_1")
        wordList = self.segSentence(words).split(" ")
        wordVec = []
        for w in wordList:
            try:
                c = model[w]
                wordVec.append(c)
            except Exception as e:
                continue
        wordVec = np.array(wordVec)
        return np.mean(wordVec,axis=0)

    # 获取评论词向量与特定词汇相似度
    def get_similarity(self,words,specialword):
        # 加载词向量模型
        model = Word2Vec.load("..\data\model_1")
        wordList = self.segSentence(words).split(" ")
        word_similarity = []
        for w in wordList:
            try:
                c = model.similarity(w,specialword)
                word_similarity.append(c)
            except Exception as e:
                continue

        return max(word_similarity)
    # 获取评论是否与某主题相关 1 相关 0 不相关
    def get_relatedSubject(self,words,specialword):
    #     加载词向量模型
        model = Word2Vec.load("..\data\model_1")
        wordList = self.segSentence(words).split(" ")
        for w in wordList:
            if w in self.subjectRelated[specialword]:
                return 1
        return 0

    # 构造评论数据集的词向量表示
    def build_vectors(self,data):
        # 数据量
        num = len(data)
        content = data["content"]
        content = content
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        content = content.apply(lambda w:self.get_vector(w))
        print(type(content))
        words_vectors = np.ones((num,100))

        for i in range(num):
            words_vectors[i,:] = content[i][:]
        print(words_vectors.shape)

        return words_vectors

    def build_subjectrain(self,dataPath):
        "构建主题训练集"
        data = pd.read_csv(dataPath)
        # 得到训练集词向量表示
        word_vectors = self.build_vectors(data)
        traindata = data[["content_id"]]
        content = data[["content"]]
        subject = data[["subject"]]
        word_vectors_dataFrame = pd.DataFrame(word_vectors)

        traindata = pd.concat([traindata,word_vectors_dataFrame],axis=1)
        # 获取评论中词汇与特定词汇的最大相似度
        for s in self.subject_eng:
            tmp = s + "_similarity"
            traindata[tmp] = 0
        traindata = pd.concat([traindata,content],axis=1)
        for (w,s) in zip(self.keywords_subject,self.subject_eng):
            tmp = s+"_similarity"
            traindata[tmp] = traindata["content"].apply(lambda x:self.get_similarity(x,w))
        for s in self.subject_eng:
            tmp = s + "_related"
            traindata[tmp] = traindata["content"].apply(lambda w:self.get_relatedSubject(w,s))

        traindata.drop(['content'],axis =1,inplace = True)
        traindata = pd.concat([traindata, subject], axis=1)
        traindata.to_csv('..\data\extractData\subjectrain_power.csv',index=None)

    def build_subjectest(self,dataPath):
        "构建主题测试集"
        data = pd.read_csv(dataPath)
        # 得到训练集词向量表示
        word_vectors = self.build_vectors(data)
        testdata = data[["content_id"]]
        content = data[["content"]]
        word_vectors_dataFrame = pd.DataFrame(word_vectors)

        testdata = pd.concat([testdata, word_vectors_dataFrame], axis=1)
        # 获取评论中词汇与特定词汇的最大相似度
        for s in self.subject_eng:
            tmp = s + "_similarity"
            testdata[tmp] = 0
        testdata = pd.concat([testdata, content], axis=1)
        for (w, s) in zip(self.keywords_subject, self.subject_eng):
            tmp = s + "_similarity"
            testdata[tmp] = testdata["content"].apply(lambda x: self.get_similarity(x, w))
        for s in self.subject_eng:
            tmp = s + "_related"
            testdata[tmp] = testdata["content"].apply(lambda w: self.get_relatedSubject(w, s))

        testdata.drop(['content'], axis=1, inplace=True)
        testdata.to_csv('../data/extractData/subjectTest.csv', index=None)

    def build_emotionTrain(self,dataPath):
        "构建情感训练集"
        pass

    def build_emotionTest(self,dataPath):
        "构建情感测试集"
        pass

if __name__ == "__main__":
    clf = Word2vec_model()
    clf.build_subjectest("../data/test_public.csv")
















