#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import data_reading
import matplotlib.pyplot as plt

poly_data , poly_keys = data_reading.readMatFile("poly_data.mat")

# we are solving a 5th order polynomial

order = 5 + 1

sampx = poly_data['sampx'][0]
sampy = poly_data['sampy']
polyx = poly_data['polyx'][0]
polyy = poly_data['polyy']
# part of painting

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sampx,sampy,color='r',linestyle='',marker='.')

# Y = [sigma x^0 * y, sigma x^1 * y ....  sigma x^k * y]
matY = []
for k in range(0,order):
  result = 0
  for index in range(0,len(sampx)):
    result = result + (sampx[index]**k) * sampy[index]
  matY.append(result)

matX = []

for k in range(0,order):
  # vector = [sigma x^k, sigma x^k+1 ....  sigma x^k+order]
  vectX = []
  for column in range(k,k+order):
    result = 0
    for x in sampx:
      result = x**column + result
    vectX.append(result)
  matX.append(vectX)

matX = np.array(matX)
print matX

# Calculate the 5th order polynomial's arguments
# for X*A = Y
# A = solve(X,Y)
matA = np.linalg.solve(matX,matY)

targety=[]
for targetx in polyx:
  result = 0
  for k in range(0,order):
    # y = a0 + a1 * x + ... ak * x^k
    result = matA[k] * (targetx**k) + result
  targety.append(result[0])

ax.plot(polyx,targety,color='b',linestyle='-',marker='')

ax.legend()
plt.show()