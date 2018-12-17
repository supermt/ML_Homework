#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import data_input
import pandas as pd
from sklearn.linear_model import Lasso

def normalize(samp_y):
  result = []
  for y in samp_y:
    if y!=0:
      y = 1
    else:
      y=0
    result.append(y)
  result = np.array(result)
  return result

def BR(train_percent):
  from sklearn.linear_model import BayesianRidge
  train_set_x,train_set_y,predict_set_x,target_y = data_input.getdata("processed.cleveland.data",train_percent)
  train_set_y = normalize(train_set_y)
  target_y = normalize(target_y)
  clf = BayesianRidge(compute_score=True)
  clf.fit(train_set_x, np.ravel(train_set_y))
  # predict_set_y = np.round(clf.predict(predict_set_x))
  predict_set_y0 = clf.predict(predict_set_x)
  predict_set_y = []
  for y in predict_set_y0:
    predict_set_y.append(int(y))
  dataframe = pd.DataFrame({'id':range(len(predict_set_y)),'predict_set_y':np.ravel(predict_set_y),'target_y':np.ravel(target_y)})
  print np.sum(np.abs(np.ravel(np.round(predict_set_y)) - target_y))
  dataframe.to_csv("./regression/BR"+str(train_percent)+".csv",index=False,sep=',')

def LASSO(train_percent):
  train_set_x,train_set_y,predict_set_x,target_y = data_input.getdata("processed.cleveland.data",train_percent)
  alphas = np.linspace(0.5,1,100)
  best_alp = 1
  best_score = 10.0
  for alp in alphas:
    LASSO = Lasso(alpha=alp, fit_intercept=False, max_iter=1000)
    LASSO.fit(train_set_x, train_set_y)
    score = LASSO.score(predict_set_x,target_y)
    if (score-1.0)**2 < (best_score-1.0)**2:
      best_alp = alp
      best_score = score
  LASSO = Lasso(alpha=best_alp, fit_intercept=False, max_iter=1000)
  LASSO.fit(train_set_x, train_set_y)
  predict_set_y = np.round(LASSO.predict(predict_set_x))
  dataframe = pd.DataFrame({'id':range(len(predict_set_y)),'predict_set_y':np.ravel(predict_set_y),'target_y':np.ravel(target_y)})
  dataframe.to_csv("./regression/LASSO"+str(train_percent)+".csv",index=False,sep=',')


if __name__ == "__main__":
  # train_percents = np.linspace(0.5,0.9,5)
  train_percents = [0.9]
  for train_percent in train_percents:
    LASSO(train_percent)
    BR(train_percent)