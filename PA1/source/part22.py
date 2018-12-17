#!/usr/bin/env python
#-*- coding:utf-8 -*-
import data_reading
from LS import do_regression as LSRegression
import matplotlib.pyplot as plt
import numpy as np
from calculating import RLS as RLSRegression
from sklearn.linear_model import Lasso
from data_reading import MeanAbsoluteError
from data_reading import MeanSquareError

def getPHI(sampx,order):
  length = len(sampx.T)
  result = np.zeros((order * len(sampx),length))
  for j in range(length):
    temp = sampx[:,j]
    for i in range(order-1):
      temp = np.vstack((temp , np.power(sampx[:,j],i + 2)))
    result[:,j] = np.ravel(temp)
  return result

if __name__ == '__main__':
  count_data , count_keys= data_reading.readMatFile("count_data.mat")

  sampx = np.matrix(count_data['trainx'])
  sampy = count_data['trainy']
  testx = np.matrix(count_data['testx'])
  testy = count_data['testy']
  order = 2

  sampx = getPHI(sampx,order)

  x_label = range(len(testx.T))

  # BR
  from sklearn.linear_model import BayesianRidge
  clf = BayesianRidge(compute_score=True)
  clf.fit(sampx.T, np.ravel(sampy))
  BR_y_prime = clf.predict(testx.T)
  ax = fig.add_subplot(n,1,4)
  ax.plot(x_label,BR_y_prime,color='r',linestyle='-',marker='',label="predict BR")
  ax.plot(x_label,testy,color='g',linestyle='',marker='.',label="real result")


  abs_errors.append(MeanAbsoluteError(BR_y_prime,testy))
  sqr_errors.append(MeanSquareError(BR_y_prime,testy))
  print "BR's absolute error is " + str(MeanAbsoluteError(BR_y_prime,testy)) 
  print "BR's Square error is " + str(MeanSquareError(BR_y_prime,testy)) 

  ax.legend()

  fig2 = plt.figure("erros")
  ax = fig2.add_subplot(121)
  ax.bar(methods,abs_errors,label="Absolute Error")
  ax.legend()

  ax = fig2.add_subplot(122)
  ax.bar(methods,sqr_errors,label="Square Error")
  ax.legend()

  plt.show()
  