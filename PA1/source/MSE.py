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

  result = np.zeros((4,5))
  for j in range(1000):
    MSEs=[]
    for percent in percents:
      try:
        mse=[]
        train_length = int(percent * total_length)
        a=np.random.randint(0,total_length,size=[1,train_length])[0]
        sampx_sub = sampx[a]
        sampy_sub = sampy[a]
        phi = grand_order(sampx_sub,order)
        # LS
        LS_y_prime = grand_order(polyx,order) * LSregression(phi.T,sampy_sub,order)
        mse.append(data_reading.MeanSquareError(LS_y_prime,polyy))
        # RLS
        RLS_target_alpha = RLS_hyper(phi.T,sampy_sub,polyx,polyy,order)
        RLS_y_prime = grand_order(polyx,order) * RLSregression(phi.T,sampy_sub,RLS_target_alpha)
        mse.append(data_reading.MeanSquareError(RLS_y_prime,polyy))
        
        #LASSO
        LASSO_target_alpha = LASSO_hyper(phi,sampy_sub,polyx,polyy,order)
        LASSO_y_prime = grand_order(polyx,order) * LASSOregression(phi,sampy_sub,LASSO_target_alpha)
        mse.append(data_reading.MeanSquareError(LASSO_y_prime,polyy))
        #RR
        RR_y_prime = grand_order(polyx,order) * RRregression(phi,sampy_sub,order)
        mse.append(data_reading.MeanSquareError(RR_y_prime,polyy))
        #BR
        sig2=5
        alpha = 0.5
        best_alpha = BR_hyper(sampx_sub,order,sampy_sub,polyx,polyy,alpha,float(sig2))
        sigma_star , BR_y_prime = BRregression(grand_order(sampx_sub,order).T,sampy_sub,polyx,polyy,order,best_alpha,float(sig2))
        mse.append(data_reading.MeanSquareError(BR_y_prime,polyy))
        MSEs.append(mse)
      except:
        pass
    MSEs = np.matrix(MSEs)
    if MSEs.shape == (4,5):
      result = result + MSEs
  plt.ylim((0, 100))
  plt.xlim((0.1, 0.8))
  plt.plot(percents,result[:,0],marker="*",ls='--',label="LS")
  plt.plot(percents,result[:,1],marker="*",ls='--',label="RLS")
  plt.plot(percents,result[:,2],marker="*",ls='--',label="LASSO")
  plt.plot(percents,result[:,3],marker="*",ls='--',label="RR")
  plt.plot(percents,result[:,4],marker="*",ls='--',label="BR")
  plt.legend()
  plt.show()