#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
from numpy import *
from sklearn import svm
import matplotlib.pyplot as plt
import data_input
import pandas as pd
from sklearn.mixture import GaussianMixture

def GMM(train_percent):
  train_set_x,train_set_y,predict_set_x,target_y = data_input.getdata("processed.cleveland.data",train_percent)
  train_set_y = ravel(train_set_y)
  target_y = ravel(target_y)
  K = len(set(train_set_y))
  Y = GaussianMixture(n_components=K, covariance_type='full').fit(train_set_x)
  
  Y = Y.predict(predict_set_x)
  label = [0,0,0,0,0]
  for k in range(K):
    print target_y[Y==k]

if __name__ == "__main__":
  # train_percents = np.linspace(0.5,0.9,5)
  # for train_percent in train_percents:
  GMM(0.9)