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
from sklearn.model_selection import train_test_split
# 程序开始时间
start_time = time.time()

# XgBoost 参数
params = {'booster': 'gbtree',
          'objective': 'rank:pairwise',
          'eval_metric': 'rmse',
          'gamma': 0.1,
          'min_child_weight': 1.1,
          'max_depth': 5,
          'lambda': 10,
          'subsample': 0.8,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.01,
          'tree_method': 'exact',
          'seed': 0,
          'nthread': 12
          }


if __name__ == '__main__':
    data = pd.read_csv("../data/extractData/subjectrain_price.csv")
    data.subject.replace("价格",1,inplace=True)
    data_x = data.drop(["content_id","subject"],axis=1)
    data_y = data.subject

    dataset = xgb.DMatrix(data_x, label=data_y)


    watchlist = [(dataset,'train')]
    model = xgb.train(params, dataset, num_boost_round=4000, evals=watchlist,early_stopping_rounds=120)
    model.save_model('../data/trained_model/xgb.model')