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
        stpwrd_dic = open(r"D:\project\DF-car_theme_emotionRecognition\data\stop_words.txt", 'r')
        stpwrd_content = stpwrd_dic.read()
        # 将停用词表转换为list
        stpwrdlst = stpwrd_content.splitlines()
        stpwrd_dic.close()
        suggest = open(r"D:\project\DF-car_theme_emotionRecognition\data\newdict.txt",'r',encoding="UTF-8")
        suggest_freq = suggest.read()
        suggest_freq = suggest_freq.splitlines()
        suggest.close()
        # 添加词典
        for sugg in suggest_freq:
            jieba.suggest_freq(sugg, True)


        self.stopwords = stpwrdlst
        self.suggest_freq = suggest_freq
        self.keywords_subject = [u"动力",u"油耗",u"价格",u"内饰",u"配置",u"安全性",u"外观",u"操控",u"空间",u"舒适性"]
        self.subject_eng = ["price","comfort","configuration","exterior","fuelConsumption","interior","manpulation","power","safety","space"]
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
            with open(r'D:\project\DF-car_theme_emotionRecognition\data\extractData\documentcut.txt', 'w',encoding="UTF-8") as f2:
                f2.write(result)
        f.close()
        f2.close()
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = LineSentence(r'D:\project\DF-car_theme_emotionRecognition\data\extractData\documentcut.txt')
        model = Word2Vec(sentences, min_count=1, seed=0, size=100,workers=1,iter=20)
        model.save(r"D:\project\DF-car_theme_emotionRecognition\data\model_1")



    # 获得单个句子的词向量表示
    def get_vector(self, words):
        # 加载词向量模型
        model = Word2Vec.load(r"D:\project\DF-car_theme_emotionRecognition\data\model_1")
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
        model = Word2Vec.load(r"D:\project\DF-car_theme_emotionRecognition\data\model_1")
        wordList = self.segSentence(words).split(" ")
        word_similarity = []
        for w in wordList:
            try:
                c = model.similarity(w,specialword)
                word_similarity.append(c)
            except Exception as e:
                continue

        return max(word_similarity)

    # 构造评论数据集的词向量表示
    def build_vectors(self,data):
        # 数据量
        num = len(data)
        content = data["content"]
        content = content[:2]
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        content = content.apply(lambda w:self.get_vector(w))
        print(type(content))
        words_vectors = np.ones((2,100))

        for i in range(2):
            words_vectors[i,:] = content[i][:]
        print(words_vectors.shape)

        return words_vectors

    # 构建训练集
    def build_train(self,dataPath):
        #    构建主题训练集
        data = pd.read_csv(dataPath)
        # 得到训练集词向量表示
        word_vectors = self.build_vectors(data)
        traindata = data[["content_id"]][:2]
        content = data[["content"]][:2]
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

        print(traindata)


    # 构建测试集
    def build_test(self,dataPath):
        pass






if __name__ == "__main__":
    clf = Word2vec_model()
    clf.build_train(r"D:\project\DF-car_theme_emotionRecognition\data\train.csv")













