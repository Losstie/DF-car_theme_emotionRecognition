#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@project: DF-car_theme_emotionRecognition
@file:xgb_model_baseline.py.py
@author: dujiahao
@create_time: 2018/9/15 17:46
@description:
"""
import xgboost as xgb
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from pylab import mpl
from sklearn.model_selection import train_test_split
# 程序开始时间
start_time = time.time()

# XgBoost 参数
params = {'booster': 'gbtree',
          'objective': 'multi:softmax',
          'eval_metric': 'rmse',
          'gamma': 0.1,
          'min_child_weight': 1.1,
          'max_depth': 7,
          'lambda': 10,
          'subsample': 0.8,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.1,
          'tree_method': 'exact',
          'seed': 0,
          'nthread': 12
          }


if __name__ == '__main__':
    price = pd.read_csv("../data/extractData/subjectrain_price.csv")
    comfort = pd.read_csv("../data/extractData/subjectrain_comfort.csv")
    configuration = pd.read_csv("../data/extractData/subjectrain_Configuration.csv")
    exterior = pd.read_csv("../data/extractData/subjectrain_Exterior.csv")
    fuelConsumption = pd.read_csv("../data/extractData/subjectrain_fuelConsumption.csv")
    interior = pd.read_csv("../data/extractData/subjectrain_Interior.csv")
    manipulation = pd.read_csv("../data/extractData/subjectrain_Manipulation.csv")
    power = pd.read_csv("../data/extractData/subjectrain_power.csv")
    safety = pd.read_csv("../data/extractData/subjectrain_safety.csv")
    space = pd.read_csv("../data/extractData/subjectrain_space.csv")

    dataset = pd.concat([price,comfort,configuration,exterior,fuelConsumption,interior,manipulation,
                         power,safety,space])
    dataset.subject.replace("舒适性",1)
    dataset.subject.replace("配置",2)
    dataset.subject.replace("")
    print(dataset.shape)
    dataset.replace(np.nan, 0,inplace=True)
    dataset.replace(np.inf, 0,inplace=True)
    dataset_x = dataset.drop(["content_id","subject"],axis=1)
    dataset_y = dataset.subject
    print(dataset_x.shape,dataset_y.shape)
    # dataset.to_csv("../data/extractData/subjectrain.csv",index=None)








    # dataset = xgb.DMatrix(dataset_x, label=dataset_y)
    #
    #
    # watchlist = [(dataset,'train')]
    # model = xgb.train(params, dataset, num_boost_round=4000, evals=watchlist,early_stopping_rounds=120)
    # model.save_model('../data/trained_model/xgb.model')

    # # 拟合模型
    clf = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=180, min_samples_split=80, max_depth=13,
                                min_samples_leaf=20, max_features=0.8)
    rf = clf.fit(dataset_x, dataset_y)
    print(clf.oob_score_)
    joblib.dump(rf, '../data/trained_model/rf.model')  # 保存训练的RF模型