#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import data_reading 
import matplotlib.pyplot as plt
from data_reading import grand_order

def do_regression(phi,sampy,order=1):  
  w = ((np.linalg.inv(phi * phi.T)) * phi) * sampy
  # part of painting
  return w

if __name__ == "__main__":
  poly_data , poly_keys = data_reading.readMatFile("poly_data.mat")

  order = 5

  sampx = poly_data['sampx'][0]
  sampy = poly_data['sampy']
  polyx = poly_data['polyx'][0]
  polyy = poly_data['polyy']

  w = do_regression(grand_order(sampx,order).T,sampy,order)
  targety = grand_order(polyx,order) * w

  fig = plt.figure("LS")
  ax = fig.add_subplot(111)
  # sample points
  ax.plot(poly_data['sampx'][0],sampy,color='r',linestyle='',marker='*',label="sample")
  # regression line
  ax.plot(polyx,targety,color='g',linestyle='-',marker='',label="predict")
  # poly points
  ax.plot(polyx,polyy,color='b',linestyle='-',marker='',label="target")
  print data_reading.MeanSquareError(targety,polyy)

  ax.legend()
  plt.show()
