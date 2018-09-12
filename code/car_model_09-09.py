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

        self.stopwords = None
        self.keywords_subject = ["动力","油耗","价格","内饰","配置","安全性","外观","操控","空间","舒适性"]
        self.keywords_emotion = [-1,0,1]
        self.positive_words = ["杠杠的","非常好","好","给力"]
        self.negative_words = ["坑","不舒服的"]

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
    def word2vec_train(self,data):
        pass

    # 获得单个句子的词向量表示
    def get_vector(self, words):
        pass

    # 构造评论数据集的词向量表示
    def build_vectors(self):
        pass



if __name__ == "__main__":
    data = pd.read_csv(r"D:\Project\DF_Car\data\train.csv")




    # 导入停用词
    # stpwrd_dic = open(r"D:\Project\DF_Car\data\stop_words.txt", 'rb')
    # stpwrd_content = stpwrd_dic.read()
    # # 将停用词表转换为list
    # stpwrdlst = stpwrd_content.splitlines()
    # stpwrd_dic.close()
    # 分词并去除空分词
    # data["jieba_result"] = data.content.apply(lambda x: ("/".join(jieba.cut(x, cut_all=True, HMM=True)).replace("//", '')))

    # model = Word2Vec(sentences,min_count=1,size=100,seed=0,workers=1)










