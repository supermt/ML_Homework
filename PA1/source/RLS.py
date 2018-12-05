#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import data_reading
import matplotlib.pyplot as plt
import calculating as reg
import random as rd
from data_reading import grand_order

def do_regression(phi,sampy,alpha):
  w = np.linalg.inv( phi * phi.T + alpha * np.eye(phi.shape[0]) ) * phi * sampy
  return w

def choose_hyper(matFi,sampy,polyx,polyy,order):
  least_error = 100
  target_alpha = 1
  alphas = np.linspace(-2,2,1000)
  for alp in alphas:
    y_prime = grand_order(polyx,order) * do_regression(matFi,sampy,alp)
    error = data_reading.MeanSquareError(y_prime,polyy)
    if error <= least_error:
      target_alpha = alp
      least_error = error
  return target_alpha

if __name__ == "__main__":
  poly_data , poly_keys = data_reading.readMatFile("poly_data.mat")

  order = 5 

  sampx = poly_data['sampx'][0]
  sampy = poly_data['sampy']
  polyx = poly_data['polyx'][0]
  polyy = poly_data['polyy']

  matFi=grand_order(sampx,order).T
  # part of painting

  fig = plt.figure("RLS")

  w = do_regression(matFi,sampy,1)
  target_alpha = choose_hyper(matFi,sampy,polyx,polyy,order)
  y_prime = grand_order(polyx,order) * do_regression(matFi,sampy,target_alpha)
  plt.plot(poly_data['sampx'][0],sampy,color='r',linestyle='',marker='*',label="sample")
  plt.plot(polyx,y_prime,color='skyblue',linestyle='-',label="lambda = "+str(target_alpha))
  plt.plot(polyx,polyy,color='g',linestyle='-',label="target")
  print "least mean error: " + str(data_reading.MeanSquareError(y_prime,polyy)) +" at " + str(target_alpha)

  plt.legend()
  plt.show()