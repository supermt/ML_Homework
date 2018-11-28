#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import itertools
import data_reading
import matplotlib.pyplot as plt
import scipy.optimize as op

def RR(X, y):
    pass

def grand_order(X,order=1):
  matX=[]
  for i in range(order+1):
      matX.append(X**i)
  result = (np.matrix(matX)).T
  return result
    

if '__main__' == __name__:
    poly_data , poly_keys = data_reading.readMatFile("poly_data.mat")

    phiT, y = grand_order(poly_data['sampx'][0],5), poly_data['sampy']
    polyx, polyy= poly_data['polyx'][0], poly_data['polyy']
    order = 5

    b = np.array(np.append(-y,y))
    D = order + 1
    n = len(y)

    f = np.array(np.append(np.zeros((1,D)),np.ones((1,n))))
    In = np.eye(n)
    matA = np.vstack((np.hstack((-phiT, -In)), np.hstack((phiT, -In))))
    th = op.linprog(f, matA, b)

    w = np.matrix(-th.x[0:D][::-1]).T
    y_prime = (grand_order(polyx,order) * w)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(poly_data['sampx'][0],y,color='r',linestyle='',marker='*',label="sample")
    # regression line
    ax.plot(polyx,y_prime,color='g',linestyle='-',marker='',label="predict")
    # poly points
    ax.plot(polyx,polyy,color='b',linestyle='',marker='.',label="target")
    ax.legend()
    plt.show()