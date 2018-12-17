#!/usr/bin/env python
#-*- coding:utf-8 -*-
import data_reading
from LS import do_regression as LSRegression
from data_reading import grand_order
import matplotlib.pyplot as plt
import numpy as np
from calculating import RLS as RLSRegression
from sklearn.linear_model import Lasso
from data_reading import MeanAbsoluteError
from data_reading import MeanSquareError

if __name__ == '__main__':
  count_data , count_keys= data_reading.readMatFile("count_data.mat")

  sampx = np.matrix(count_data['trainx'])
  sampy = count_data['trainy']
  testx = np.matrix(count_data['testx'])
  testy = count_data['testy']
  order = 1

  x_label = range(len(testx.T))


  fig = plt.figure("plots")
  
  methods = ["LS","RLS","LASSO","BR"]
  abs_errors = []
  sqr_errors = []

  n = len(methods)
  # LS
  LS_result = LSRegression(sampx,sampy)
  LS_y_prime = np.round(testx.T * LS_result)
  ax = fig.add_subplot(n,1,1)
  ax.plot(x_label,LS_y_prime,color='r',linestyle='-',marker='',label="predict LS")
  ax.plot(x_label,testy,color='g',linestyle='',marker='.',label="real result")

  abs_errors.append(MeanAbsoluteError(LS_y_prime,np.round(testy)))
  sqr_errors.append(MeanSquareError(LS_y_prime,np.round(testy)))
  print "LS's absolute error is " + str(MeanAbsoluteError(LS_y_prime,np.round(testy))) 
  print "LS's Square error is " + str(MeanSquareError(LS_y_prime,np.round(testy))) 
  ax.legend()

  #RLS
  alpha = 1
  RLS_result = RLSRegression(sampx,sampy,alpha)
  RLS_y_prime = np.round(testx.T * RLS_result) 
  ax = fig.add_subplot(n,1,2)
  ax.plot(x_label,RLS_y_prime,color='r',linestyle='-',marker='',label="predict RLS")
  ax.plot(x_label,testy,color='g',linestyle='',marker='.',label="real result")

  abs_errors.append(MeanAbsoluteError(RLS_y_prime,np.round(testy)))
  sqr_errors.append(MeanSquareError(RLS_y_prime,np.round(testy)))
  print "RLS's absolute error is " + str(MeanAbsoluteError(RLS_y_prime,np.round(testy))) 
  print "RLS's Square error is " + str(MeanSquareError(RLS_y_prime,np.round(testy))) 
  ax.legend()

  #LASSO
  alphas = np.linspace(0,1,100)
  best_alp = 1
  best_score = 10.0
  for alp in alphas:
    LASSO = Lasso(alpha=alp, fit_intercept=False, max_iter=1000)
    LASSO.fit(sampx.T, sampy)
    score = LASSO.score(testx.T,testy)
    if (score-1.0)**2 < (best_score-1.0)**2:
      best_alp = alp
      best_score = score

  LASSO = Lasso(alpha=best_alp, fit_intercept=False, max_iter=1000)
  LASSO.fit(sampx.T, sampy)
  LASSO_y_prime = LASSO.predict(testx.T)

  ax = fig.add_subplot(n,1,3)
  ax.plot(x_label,LASSO_y_prime,color='r',linestyle='-',marker='',label="predict LASSO")
  ax.plot(x_label,testy,color='g',linestyle='',marker='.',label="real result")

  abs_errors.append(MeanAbsoluteError(LASSO_y_prime,testy))
  sqr_errors.append(MeanSquareError(LASSO_y_prime,testy))
  print "LASSO's absolute error is " + str(MeanAbsoluteError(LASSO_y_prime,np.round(testy))) 
  print "LASSO's Square error is " + str(MeanSquareError(LASSO_y_prime,np.round(testy))) 

  ax.legend()


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
  print "BR's absolute error is " + str(MeanAbsoluteError(BR_y_prime,np.round(testy))) 
  print "BR's Square error is " + str(MeanSquareError(BR_y_prime,np.round(testy))) 

  ax.legend()

  fig2 = plt.figure("erros")
  ax = fig2.add_subplot(121)
  ax.bar(methods,abs_errors,label="Absolute Error")
  ax.legend()

  ax = fig2.add_subplot(122)
  ax.bar(methods,sqr_errors,label="Square Error")
  ax.legend()

  plt.show()
  