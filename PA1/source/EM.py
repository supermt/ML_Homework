#!/usr/bin/env python
#-*- coding:utf-8 -*-
import matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt

# 根据x输出正态分布的f(x)
def gaussian_distribution(x, mu,  standard_dev):
    return 1/math.sqrt(2*math.pi*(standard_dev**2)) * math.exp(-((x - mu)**2) / (2*standard_dev**2))

def poisson_distribution(x,lambda_arg):
  return (1 / math.factorial(x)) * math.exp( -(lambda_arg) ) * lambda_arg**x
  

# E step: 根据分布参数计算样本是男生身高的概率，这个是隐变量
def E_step(heights, mu_boy, variance_boy, mu_girl, variance_girl):
    proba_of_boys = []
    for i in heights:
        b = gaussian_distribution(i, mu_boy, variance_boy)
        g = gaussian_distribution(i, mu_girl, variance_girl)
        proba_of_boys.append(b/(b+g))
    return np.asarray(proba_of_boys)

# M step: 现在已知隐变量的概率分布，重新计算男生、女生的概率分布
def M_step(heights, proba_of_boys):
    mu_boy = (heights * proba_of_boys).sum() / proba_of_boys.sum()
    variance_boy = math.sqrt((proba_of_boys * (heights - mu_boy)**2).sum() / proba_of_boys.sum())
    
    proba_of_girl = 1 - proba_of_boys
    mu_girl = (heights * proba_of_girl).sum() / proba_of_girl.sum()
    variance_girl = math.sqrt((proba_of_girl * (heights - mu_girl)**2).sum() / proba_of_girl.sum())
    return (mu_boy, variance_boy, mu_girl, variance_girl)

# EM算法的完整流程
# 1.随机初始化分布参数，2. E step 3. M step 4.反复迭代直到参数基本不怎么变化
def EM_iteration(heights, iters = 5):
    mu_boy = 180
    variance_boy = 5
    mu_girl = 150
    variance_girl = 5
    
    for i in range(iters):
        probs = E_step(heights, mu_boy, variance_boy, mu_girl, variance_girl)
        mu_boy, variance_boy, mu_girl, variance_girl = M_step(heights, probs)
        print ("mu_boy: %f, variance_boy: %f, mu_girl: %f, variance_girl: %f" 
               % (mu_boy, variance_boy, mu_girl, variance_girl))

# 随机产生男生和女生的身高数据，男生100，女生100
# 其中男生的身高均值为175，标准差为10，女生为160，标准差为10
np.random.seed(19680801)

# example data
mu = 175  # mean of distribution
sigma = 10  # standard deviation of distribution
height_of_boy = mu + sigma * np.random.randn(100)

# example data
mu = 160  # mean of distribution
sigma = 10  # standard deviation of distribution
height_of_girl = mu + sigma * np.random.randn(100)

# 将数据混合在一起
mixed_height = np.asarray([height_of_boy,height_of_girl])
mixed_height = mixed_height.reshape(200,)

EM_iteration(mixed_height)


print poisson_distribution(1,0.929)