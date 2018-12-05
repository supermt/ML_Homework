#!/usr/bin/env python
#-*- coding:utf-8 -*-
import scipy.io as scio
import numpy as np

def readMatFile(filename):
  datasets = scio.loadmat(filename);
  datakeys = []
  for data_key in datasets:
    if (data_key[0] != '_'):
      datakeys.append(data_key)

  return datasets , datakeys

def grand_order(X,order=1):
    matX=[]
    for i in range(order+1):
        matX.append(X**i)
    result = (np.matrix(matX)).T
    return result

def MeanSquareError(y_prime, polyy):        # mean-square error
    # y_prime = np.array(y_prime)
    # sum = 0
    # n = len(polyy)
    # for i in range(n):
    #     error = y_prime[i]-polyy[i]
    #     sum += (error**2)
    # avr = sum/n
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_prime, polyy)
    return mse
    
def MeanAbsoluteError(y_prime, polyy):        # mean-absolute error
    y_prime = np.array(y_prime)
    sum = 0
    n = len(polyy)
    for i in range(n):
        error = y_prime[i]-polyy[i]
        sum += (abs(error))
    avr = sum/n
    return avr[0]

def main():
  count_data , count_keys= readMatFile("count_data.mat")
  poly_data , poly_keys = readMatFile("poly_data.mat")
  
if __name__ == '__main__':
  main()