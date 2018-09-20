#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@project: DF-car_theme_emotionRecognition
@file:lr_emtion_baseline.py.py
@author: dujiahao
@create_time: 2018/9/17 22:07
@description:
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

def settingRules():
    "设定规则"
    pass


if __name__ == "__main__":
    """
        训练模型
        """
    price = pd.read_csv("../data/extractData/emotiontrain_price.csv")
    comfort = pd.read_csv("../data/extractData/emotiontrain_comfort.csv")
    configuration = pd.read_csv("../data/extractData/emotiontrain_Configuration.csv")
    exterior = pd.read_csv("../data/extractData/emotiontrain_Exterior.csv")
    fuelConsumption = pd.read_csv("../data/extractData/emotiontrain_fuelConsumption.csv")
    interior = pd.read_csv("../data/extractData/emotiontrain_Interior.csv")
    manipulation = pd.read_csv("../data/extractData/emotiontrain_Manipulation.csv")
    power = pd.read_csv("../data/extractData/emotiontrain_power.csv")
    safety = pd.read_csv("../data/extractData/emotiontrain_safety.csv")
    space = pd.read_csv("../data/extractData/emotiontrain_space.csv")

    dataset = pd.concat([price,comfort,configuration,exterior,fuelConsumption,interior,manipulation,
                         power,safety,space])
    dataset.replace(np.nan, 0,inplace=True)
    dataset.replace(np.inf, 0,inplace=True)

    dataset_x = dataset.drop(["content_id","sentiment_value","sentiment_word"],axis=1)
    dataset_y = dataset.sentiment_value
    print(dataset.shape)
    print(dataset_x.shape, dataset_y.shape)

    lr = LogisticRegression(C=1.0, penalty='l2', solver='newton-cg', class_weight='balanced')
    lr.fit(dataset_x,dataset_y)
    joblib.dump(lr, '../data/trained_emotion_model/lr_emotion_baseline.model')

    # 预测
    testdata = pd.read_csv("../data/extractData/emotionTest.csv")
    print(testdata.shape)
    testdata_x = testdata.drop(["content_id","sentiment_word"],axis=1)
    lr = joblib.load('../data/trained_emotion_model/lr_emotion_baseline.model')
    result = lr.predict(testdata_x)
    result = pd.DataFrame(result)
    testdata["sentiment_value"] = result.copy()
    testdata.sentiment_value.apply(lambda s:int(s))

    # 合并主题和情感
    subject = pd.read_csv("../data/predict/predict_subject_baseline.csv")
    subject["sentiment_value"] = testdata["sentiment_value"]
    subject["sentiment_word"] = testdata["sentiment_word"]
    predict = subject
    predict.to_csv("../data/predict/predictBaseline.csv",index=None)


