#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import itertools
import data_reading
import matplotlib.pyplot as plt
from data_reading import grand_order

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
    

if '__main__' == __name__:
    poly_data , poly_keys = data_reading.readMatFile("poly_data.mat")
    order = 5

    X, y = grand_order(poly_data['sampx'][0],order), poly_data['sampy']
    polyx, polyy= poly_data['polyx'][0], poly_data['polyy']
    count_data , count_keys= data_reading.readMatFile("count_data.mat")

    w = lasso_regression(X, y, lambd=10)
    y_prime = (grand_order(polyx,order) * w)
    
    least_error = 100
    target_alpha = 1
    alphas = np.linspace(-2,2,100)
    for alp in alphas:
        w = lasso_regression(X, y, alp)
        y_prime = (grand_order(polyx,order) * w)
        error = data_reading.MeanSquareError(y_prime,polyy)
        if error <= least_error:
            target_alpha = alp
            least_error = error
    
    w = lasso_regression(X,y,target_alpha)
    y_prime = grand_order(polyx,order) * w
    
    fig = plt.figure("LASSO")
    ax = fig.add_subplot(111)
    ax.plot(poly_data['sampx'][0],y,color='r',linestyle='',marker='*',label="sample")
    # regression line
    # ax.plot(polyx,y_prime,color='g',linestyle='-',marker='',label="predict")
    ax.plot(polyx,y_prime,color='g',linestyle='-',label="lambda = "+str(target_alpha))
    print "least mean error: " + str(data_reading.MeanSquareError(y_prime,polyy)) +" at " + str(target_alpha)

    # poly points
    ax.plot(polyx,polyy,color='b',linestyle='-',marker='',label="target")
    ax.legend()
    plt.show()
