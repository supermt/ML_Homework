#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import itertools
import data_reading
import matplotlib.pyplot as plt
from data_reading import grand_order

def do_regression(phi, y, lambd=0.2, threshold=0.1):
    rss = lambda phi, y, w: (y - phi*w).T*(y - phi*w)
    m,n = phi.shape
    w = np.matrix(np.zeros((n, 1)))
    r = rss(phi, y, w)
    niter = itertools.count(1)
    for it in niter:
        for k in range(n):
            z_k = (phi[:, k].T*phi[:, k])[0, 0]
            p_k = 0
            for i in range(m):
                p_k += phi[i, k]*(y[i, 0] - sum([phi[i, j]*w[j, 0] for j in range(n) if j != k]))
            if p_k < -lambd/2:
                w_k = (p_k + lambd/2)/z_k
            elif p_k > lambd/2:
                w_k = (p_k - lambd/2)/z_k
            else:
                w_k = 0
            w[k, 0] = w_k
        r_prime = rss(phi, y, w)
        delta = abs(r_prime - r)[0, 0]
        r = r_prime

        if delta < threshold:
            break
    return w
    
def choose_hyper(phi,y,polyx,polyy,order):
    w = do_regression(phi, y, lambd=10)
    y_prime = (grand_order(polyx,order) * w)
    
    least_error = 100
    target_alpha = 1
    alphas = np.linspace(-20,-18,10)
    for alp in alphas:
        w = do_regression(phi, y, alp)
        y_prime = (grand_order(polyx,order) * w)
        error = data_reading.MeanSquareError(y_prime,polyy)
        if error <= least_error:
            target_alpha = alp
            least_error = error
    return target_alpha

if '__main__' == __name__:
    poly_data , poly_keys = data_reading.readMatFile("poly_data.mat")
    order = 10

    X, y = grand_order(poly_data['sampx'][0],order), poly_data['sampy']
    polyx, polyy= poly_data['polyx'][0], poly_data['polyy']
    count_data , count_keys= data_reading.readMatFile("count_data.mat")
    target_alpha = choose_hyper(X,y,polyx,polyy,order)
    w = do_regression(X,y,target_alpha)
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
