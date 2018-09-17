#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@project: DF-car_theme_emotionRecognition
@file:rf_subject_model_baseline.py.py
@author: dujiahao
@create_time: 2018/9/15 17:46
@description:
"""
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from pylab import mpl
from sklearn.model_selection import train_test_split
# 程序开始时间
start_time = time.time()


if __name__ == '__main__':
    """
    训练模型
    """
    # price = pd.read_csv("../data/extractData/subjectrain_price.csv")
    # comfort = pd.read_csv("../data/extractData/subjectrain_comfort.csv")
    # configuration = pd.read_csv("../data/extractData/subjectrain_Configuration.csv")
    # exterior = pd.read_csv("../data/extractData/subjectrain_Exterior.csv")
    # fuelConsumption = pd.read_csv("../data/extractData/subjectrain_fuelConsumption.csv")
    # interior = pd.read_csv("../data/extractData/subjectrain_Interior.csv")
    # manipulation = pd.read_csv("../data/extractData/subjectrain_Manipulation.csv")
    # power = pd.read_csv("../data/extractData/subjectrain_power.csv")
    # safety = pd.read_csv("../data/extractData/subjectrain_safety.csv")
    # space = pd.read_csv("../data/extractData/subjectrain_space.csv")
    #
    # dataset = pd.concat([price,comfort,configuration,exterior,fuelConsumption,interior,manipulation,
    #                      power,safety,space])
    #
    # print(dataset.shape)
    # dataset.replace(np.nan, 0,inplace=True)
    # dataset.replace(np.inf, 0,inplace=True)
    # dataset.subject = dataset.subject.apply(lambda w:"其他" if w!="价格" else w)
    # dataset.subject = dataset.subject.apply(lambda w:"其他" if w!="动力" else w)
    # dataset.subject = dataset.subject.apply(lambda w:"其他" if w!="油耗" else w)
    # dataset.subject = dataset.subject.apply(lambda w:"其他" if w!="内饰" else w)
    # dataset.subject = dataset.subject.apply(lambda w:"其他" if w!="配置" else w)
    # dataset.subject = dataset.subject.apply(lambda w:"其他" if w!="安全性" else w)
    # dataset.subject = dataset.subject.apply(lambda w:"其他" if w!="外观" else w)
    # dataset.subject = dataset.subject.apply(lambda w:"其他" if w!="操控" else w)
    # dataset.subject = dataset.subject.apply(lambda w:"其他" if w!="空间" else w)
    # dataset.subject = dataset.subject.apply(lambda w:"其他" if w!="舒适性" else w)
    #
    #
    # dataset_x = dataset.drop(["content_id","subject"],axis=1)
    # dataset_y = dataset.subject
    # print(dataset_x.shape,dataset_y.shape)
    # print(dataset_y)

    # # 拟合模型
    # clf = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=180, min_samples_split=80, max_depth=13,
    #                             min_samples_leaf=20, max_features=0.8)
    # rf = clf.fit(dataset_x, dataset_y)
    # print(clf.oob_score_)
    # # 保存训练的RF模型
    # joblib.dump(rf, '../data/trained_model/rf_comfort2.model')
    """
    预测部分
    """
    # 载入训练好的模型
    # model = joblib.load('../data/trained_model/rf_space2.model')
    # # 预测
    # testdata = pd.read_csv("../data/extractData/subjectTest.csv")
    # test_public = pd.read_csv("../data/test_public.csv")
    # testdata_x = testdata.drop(["content_id"],axis=1)
    # result = model.predict(testdata_x)
    # result = pd.DataFrame(result)
    # test_public["subject"] = result.copy()
    # test_public.to_csv("../data/predict/space2.csv",index=None)

    """
    拼接
    """
    comfort = pd.read_csv("../data/predict/comfort2.csv")
    comfort = comfort[comfort.subject == "舒适性"]
    configuration = pd.read_csv("../data/predict/configuration2.csv")
    configuration = configuration[configuration.subject == "配置"]
    exterior = pd.read_csv("../data/predict/exterior2.csv")
    exterior = exterior[exterior.subject == "配置"]
    fuelConsumption = pd.read_csv("../data/predict/fuelConsumption2.csv")
    fuelConsumption = fuelConsumption[fuelConsumption.subject=="油耗"]
    interior = pd.read_csv("../data/predict/interior2.csv")
    manpulation = pd.read_csv("../data/predict/manpulation2.csv")
    manpulation = manpulation[manpulation.subject == "操控"]
    power = pd.read_csv("../data/predict/power2.csv")
    power = power[power.subject == "动力"]
    price = pd.read_csv("../data/predict/price2.csv")
    price = price[price.subject=="价格"]
    safety = pd.read_csv("../data/predict/safety2.csv")
    safety = safety[safety.subject=="安全性"]
    space = pd.read_csv("../data/predict/space2.csv")
    space = space[space.subject=="空间"]
    predict = pd.concat([price, comfort, configuration, exterior, fuelConsumption, interior, manpulation,
                         power,safety,space])

    predict.to_csv("../data/predict/predict_baseline.csv",index=None)


