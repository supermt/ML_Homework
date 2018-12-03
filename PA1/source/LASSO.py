#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import itertools
import data_reading
import matplotlib.pyplot as plt

def lasso_regression(X, y, lambd=0.2, threshold=0.1):
    rss = lambda X, y, w: (y - X*w).T*(y - X*w)
    m,n = X.shape
    w = np.matrix(np.zeros((n, 1)))
    r = rss(X, y, w)
    niter = itertools.count(1)
    for it in niter:
        for k in range(n):
            z_k = (X[:, k].T*X[:, k])[0, 0]
            p_k = 0
            for i in range(m):
                p_k += X[i, k]*(y[i, 0] - sum([X[i, j]*w[j, 0] for j in range(n) if j != k]))
            if p_k < -lambd/2:
                w_k = (p_k + lambd/2)/z_k
            elif p_k > lambd/2:
                w_k = (p_k - lambd/2)/z_k
            else:
                w_k = 0
            w[k, 0] = w_k
        r_prime = rss(X, y, w)
        delta = abs(r_prime - r)[0, 0]
        r = r_prime

        if delta < threshold:
            break
    return w

def grand_order(X,order=1):
    matX=[]
    for i in range(order+1):
        matX.append(X**i)
    result = (np.matrix(matX)).T
    return result
    

if '__main__' == __name__:
    poly_data , poly_keys = data_reading.readMatFile("poly_data.mat")

    X, y = grand_order(poly_data['sampx'][0],5), poly_data['sampy']
    polyx, polyy= poly_data['polyx'][0], poly_data['polyy']

    order = 5

    w = lasso_regression(X, y, lambd=10)
    y_prime = (grand_order(polyx,order) * w)
    
    fig = plt.figure("LASSO")
    ax = fig.add_subplot(111)
    ax.plot(poly_data['sampx'][0],y,color='r',linestyle='',marker='*',label="sample")
    # regression line
    ax.plot(polyx,y_prime,color='g',linestyle='-',marker='',label="predict")
    # poly points
    ax.plot(polyx,polyy,color='b',linestyle='-',marker='',label="target")
    ax.legend()
    plt.show()