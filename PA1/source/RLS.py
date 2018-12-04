#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import data_reading
import matplotlib.pyplot as plt
import calculating as reg
import random as rd

poly_data , poly_keys = data_reading.readMatFile("poly_data.mat")

# we are solving a 5th order polynomial and we should 
# however create a k + 1 matrix, for we are counting
# from 0

order = 5 + 1

sampx = poly_data['sampx'][0]
sampy = poly_data['sampy']
polyx = poly_data['polyx'][0]
polyy = poly_data['polyy']

matFi=[]
# matFi = [[1,1 .... 1]
#          [x1,x2 .... xn]
#           ....
#          [x1^k,x2^k .... xn^k]]
for k in range(0,order):
  vectfi = []
  for x in sampx:
    vectfi.append(x**k)
  matFi.append(vectfi)
matFi = np.array(matFi)

# part of painting

fig = plt.figure()

subfigs = []

# for RLS ,testing lambda 1 and 1000
RLS = fig.add_subplot(111)
RLS.plot(sampx,sampy,color='r',linestyle='',marker='.')
RLS.plot(polyx,polyy,color='g',linestyle='-',label="target")
y_prime = reg.verify(polyx,reg.RLS(matFi,sampy,1),order)
RLS.plot(polyx,y_prime,color='blue',linestyle='-',label="lambda = 1")
y_prime = reg.verify(polyx,reg.RLS(matFi,sampy,1000),order)

least_error = 100
target_alpha = 1
alphas = np.linspace(-2,2,1000)
for alp in alphas:
  y_prime = reg.verify(polyx,reg.RLS(matFi,sampy,alp),order)
  error = data_reading.MeanSquareError(y_prime,polyy)
  if error <= least_error:
    target_alpha = alp
    least_error = error
    
y_prime = reg.verify(polyx,reg.RLS(matFi,sampy,target_alpha),order)
RLS.plot(polyx,y_prime,color='skyblue',linestyle='-',label="lambda = "+str(target_alpha))
print "least mean error: " + str(data_reading.MeanSquareError(y_prime,polyy)) +" at " + str(target_alpha)

RLS.legend()
plt.show()