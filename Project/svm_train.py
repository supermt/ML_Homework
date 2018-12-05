#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import data_input
import pandas as pd

def svm_training(train_percent):
  train_set_x,train_set_y,predict_set_x,target_y = data_input.getdata("processed.cleveland.data",train_percent)
  clf = svm.SVC(gamma='scale')
  clf.fit(train_set_x,np.ravel(train_set_y))
  predict_set_y = clf.predict(predict_set_x)
  dataframe = pd.DataFrame({'id':range(len(predict_set_y)),'predict_set_y':np.ravel(predict_set_y),'target_y':np.ravel(target_y)})
  dataframe.to_csv("./svm/"+str(train_percent)+".csv",index=False,sep=',')

def svm_training_LinearSVC(train_percent):
  train_set_x,train_set_y,predict_set_x,target_y = data_input.getdata("processed.cleveland.data",train_percent)
  clf = svm.LinearSVC()
  clf.fit(train_set_x,np.ravel(train_set_y))
  predict_set_y = clf.predict(predict_set_x)
  dataframe = pd.DataFrame({'id':range(len(predict_set_y)),'predict_set_y':np.ravel(predict_set_y),'target_y':np.ravel(target_y)})
  dataframe.to_csv("./svm/linear"+str(train_percent)+".csv",index=False,sep=',')

def svm_training_SVR(train_percent):
  train_set_x,train_set_y,predict_set_x,target_y = data_input.getdata("processed.cleveland.data",train_percent)
  clf = svm.SVR()
  clf.fit(train_set_x,np.ravel(train_set_y))
  predict_set_y = clf.predict(predict_set_x)
  dataframe = pd.DataFrame({'id':range(len(predict_set_y)),'predict_set_y':np.ravel(predict_set_y),'target_y':np.ravel(target_y)})
  dataframe.to_csv("./svm/svr"+str(train_percent)+".csv",index=False,sep=',')
  

attributes = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]

if __name__ == "__main__":
  train_percents = np.linspace(0.8,0.9,10)
  for train_percent in train_percents:
    svm_training(train_percent)
    svm_training_LinearSVC(train_percent)
