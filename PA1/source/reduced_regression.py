#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import data_reading 
import matplotlib.pyplot as plt
from data_reading import grand_order
from LS import do_regression as LSregression
from RLS import do_regression as RLSregression
from RLS import choose_hyper as RLS_hyper
from LASSO import do_regression as LASSOregression
from LASSO import choose_hyper as LASSO_hyper
from RR import do_regression as RRregression
from BR import do_regression as BRregression
from BR import choose_hyper as BR_hyper

if __name__ == "__main__":
  poly_data , poly_keys = data_reading.readMatFile("poly_data.mat")
  order = 5
  sampx = poly_data['sampx'][0]
  sampy = poly_data['sampy']
  polyx = poly_data['polyx'][0]
  polyy = poly_data['polyy']

  total_length = len(sampx)
  percents = [0.1, 0.25, 0.5, 0.75]

  for percent in percents:
    fig = plt.figure(str(percent*100)+"%percent dataset")
    BR = plt.subplot(3,1,1)
    BR.set_title("BR")
    LS = plt.subplot(3,2,3)
    LS.set_title("LR")
    RLS = plt.subplot(3,2,4)
    RLS.set_title("RLS")
    LASSO = plt.subplot(3,2,6)
    LASSO.set_title("LASSO")
    RR = plt.subplot(3,2,5)
    RR.set_title("RR")
    LS.plot(polyx,polyy,color='b',linestyle='-',marker='',label="target")
    RLS.plot(polyx,polyy,color='b',linestyle='-',marker='',label="target")
    LASSO.plot(polyx,polyy,color='b',linestyle='-',marker='',label="target")
    RR.plot(polyx,polyy,color='b',linestyle='-',marker='',label="target")
    
    train_length = int(percent * total_length)
    sampx_sub = sampx[range(train_length)]
    sampy_sub = sampy[range(train_length)]
    phi = grand_order(sampx_sub,order)
    # LS
    LS_y_prime = grand_order(polyx,order) * LSregression(phi.T,sampy_sub,order)
    LS.plot(polyx,LS_y_prime,color='r',linestyle='-',marker='',label="predict")
    # RLS
    RLS_target_alpha = RLS_hyper(phi.T,sampy_sub,polyx,polyy,order)
    RLS_y_prime = grand_order(polyx,order) * RLSregression(phi.T,sampy_sub,RLS_target_alpha)
    RLS.plot(polyx,RLS_y_prime,color='r',linestyle='-',marker='',label="predict")
    #LASSO
    LASSO_target_alpha = LASSO_hyper(phi,sampy_sub,polyx,polyy,order)
    LASSO_y_prime = grand_order(polyx,order) * LASSOregression(phi,sampy_sub,LASSO_target_alpha)
    LASSO.plot(polyx,LASSO_y_prime,color='r',linestyle='-',marker='',label="predict")
    #RR
    RR_y_prime = grand_order(polyx,order) * RRregression(phi,sampy_sub,order)
    RR.plot(polyx,RR_y_prime,color='r',linestyle='-',marker='',label="predict")
    #BR
    sig2=5
    alpha = 0.5
    best_alpha = BR_hyper(sampx_sub,order,sampy_sub,polyx,polyy,alpha,float(sig2))
    sigma_star , BR_y_prime = BRregression(grand_order(sampx_sub,order).T,sampy_sub,polyx,polyy,order,best_alpha,float(sig2))
    BR.plot(polyx,BR_y_prime,color='r',linestyle='-',marker='',label="predict")
    BR.errorbar(polyx,polyy,color='b', yerr=np.round(sigma_star), xlolims=True,label="standard deviation around the mean")

    RR.legend()
    BR.legend()
    RLS.legend()
    LS.legend()
    LASSO.legend()

  plt.show()