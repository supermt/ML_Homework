#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import data_reading
import matplotlib.pyplot as plt
import calculating as reg
import random as rd
from numpy import *
# Regularized Least-squares regression

def grand_order(X,order=1):
    matX=[]
    for i in range(order+1):
        matX.append(X**i)
    result = (np.matrix(matX))
    return result

def BR_solve( phi, y, alpha = 0.25, sig2 = 5.0):
    tmp1 = 1.0 / alpha * np.eye(phi.shape[0])
    tmp2 = 1.0 / sig2 * phi * phi.T
    sigma = np.linalg.inv(tmp1 + tmp2)
    miu = 1.0 / sig2 * sigma * phi * y
    return miu, sigma

def BR_pred(phi, miu, sigma):
    length = phi.shape[1]
    sig2_star = zeros(length)
    for i in range(length):
        sig2_star[i] = ((phi.T[i]) * sigma * (phi.T[i]).T)[0,0]
    miu_star = phi.T * miu
    return miu_star, sig2_star

def do_regression(phi,sampy,polyx,polyy,order,alpha,sig2):
    #compute parameter theta
    miu_hat, sigma_hat = BR_solve(phi,sampy,alpha,sig2)
    # predict
    miu_star, sig2_pred = BR_pred(grand_order(polyx,order),miu_hat,sigma_hat)
    return sig2_pred, miu_star

def choose_hyper(sampx,order,sampy,polyx,polyy,alpha,sig2):
    sigma_star , y_prime = do_regression(grand_order(sampx,order),sampy,polyx,polyy,order,alpha,float(sig2))
    best_alpha=alpha
    least_mean_error = data_reading.MeanSquareError(y_prime,polyy)
    alphas = linspace(0.1,11,100)
    for alp in alphas:
        sigma_star , y_prime = do_regression(grand_order(sampx,order),sampy,polyx,polyy,order,alp,float(sig2))
        mean_error = data_reading.MeanSquareError(y_prime,polyy)
        if mean_error<=least_mean_error:
            least_mean_error = mean_error
            best_alpha = alp
    return best_alpha

if '__main__' == __name__:
    poly_data , poly_keys = data_reading.readMatFile("poly_data.mat")

    order=5
    alpha=0.5
    
    #initialization
    sampx = poly_data['sampx'][0]
    sampy = poly_data['sampy']
    polyx, polyy= poly_data['polyx'][0], poly_data['polyy']
    
    sig2=5.0 # noise
    best_alpha = choose_hyper(sampx,order,sampy,polyx,polyy,alpha,float(sig2))
    sigma_star , y_prime = do_regression(grand_order(sampx,order),sampy,polyx,polyy,order,best_alpha,float(sig2))
    print "least mean error: " + str(data_reading.MeanSquareError(y_prime,polyy)) +" at " + str(best_alpha)

    # paint
    fig = plt.figure('Bayesian Regression')
    ax = fig.add_subplot(111)
    ax.plot(polyx,y_prime,color='g',linestyle='-',marker='',label="prediction with least mean error when alpha is "+str(round(best_alpha,5)))
    ax.errorbar(polyx,polyy,color='r', yerr=np.round(sigma_star), xlolims=True,label="standard deviation around the mean")
    ax.legend()
    plt.show()