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

def BR(sampx,sampy,polyx,polyy,order,alpha,sig2):
    phi = grand_order(sampx,order)
    length = len(polyx)
    phii=zeros((order + 1,length))
    predy=zeros((length,1))
    I=eye(order+1)
    #compute parameter theta
    
    sigma=linalg.inv((1/float(alpha))*I+(1/float(sig2))*(phi*phi.T))
    mu=(1/float(sig2)) * sigma * phi * sampy
    mu_star = zeros(length)
    sigma_star = zeros(length)
    py = zeros(length)
    # predict 
    for i in range(length):
        for j in range(order+1):
            phii[j][i]=polyx[i]**j
        mu_star[i]=phii[:,i].T * mu
        sigma_star[i]= phii[:,i] * sigma * phii[:,i].reshape(-1,1) # phii[:,i]) is an array and use reshape to transfer into a column
        exp_term = (-(polyx[i]-mu_star[i])**2)/((2*sigma_star[i]))
        py[i]=1/sqrt(2*pi*sigma_star[i])*exp(exp_term)
        predy[i]=mu_star[i]

    return sigma_star , predy

if '__main__' == __name__:
    poly_data , poly_keys = data_reading.readMatFile("poly_data.mat")

    order=5
    alpha=5
    
    #initialization
    sampx = poly_data['sampx'][0]
    sampy = poly_data['sampy']
    polyx, polyy= poly_data['polyx'][0], poly_data['polyy']
    
    sig2=5.0 # noise

    sigma_star , y_prime = BR(sampx,sampy,polyx,polyy,order,alpha,float(sig2))
    
    best_alpha=alpha
    least_mean_error = data_reading.MeanSquareError(y_prime,polyy)
    alphas = linspace(10,11,100)
    for alp in alphas:
        sigma_star , y_prime = BR(sampx,sampy,polyx,polyy,order,alp,float(sig2))
        mean_error = data_reading.MeanSquareError(y_prime,polyy)
        if mean_error<=least_mean_error:
            least_mean_error = mean_error
            best_alpha = alp

    sigma_star , y_prime = BR(sampx,sampy,polyx,polyy,order,best_alpha,float(sig2))
    print "least mean error: " + str(data_reading.MeanSquareError(y_prime,polyy)) +" at " + str(best_alpha)

    # paint
    fig = plt.figure('Bayesian Regression')
    ax = fig.add_subplot(111)
    ax.plot(polyx,y_prime,color='g',linestyle='-',marker='',label="prediction with least mean error when alpha is "+str(round(best_alpha,5)))
    ax.errorbar(polyx,polyy,color='r', yerr=sigma_star, xlolims=True,label="standard deviation around the mean")
    ax.legend()
    plt.show()