#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import data_reading

MAX_ITER=100

def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)

def getExpectation(Y, mu, cov, alpha):
    N = Y.shape[0]
    K = alpha.shape[0]

    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"

    gamma = np.mat(np.zeros((N, K)))

    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma

def maximize(Y, gamma):
    N, D = Y.shape
    K = gamma.shape[1]

    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)

    for k in range(K):
        Nk = np.sum(gamma[:, k])
        for d in range(D):
            mu[k, d] = np.sum(np.multiply(gamma[:, k], Y[:, d])) / Nk

        cov_k = np.mat(np.zeros((D, D)))
        for i in range(N):
            cov_k += gamma[i, k] * (Y[i] - mu[k]).T * (Y[i] - mu[k]) / Nk
        cov.append(cov_k)

        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha


def scale_data(Y):
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    return Y

def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    return mu, cov, alpha


def GMM_EM(Y, K, times):
    Y = scale_data(Y)
    mu, cov, alpha = init_params(Y.shape, K)
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    return mu, cov, alpha

def do_the_clustering(datapoints,n):
  points = datapoints.T
  matY = np.matrix(points, copy=True)
  K = 4
  plt.subplot(2,2,n)
  # for i in range(4):
    # plt.scatter((points[label == i])[:,0],(points[label == i])[:,1])
  mu, cov, alpha = GMM_EM(matY, K, MAX_ITER)
  N = points.shape[0]
  gamma = getExpectation(matY, mu, cov, alpha)
  label = np.array(gamma.argmax(axis=1).flatten().tolist()[0])

  for i in range(K):
    plt.scatter((points[label == i])[:,0],(points[label == i])[:,1])


if __name__ == '__main__':
  data = data_reading.main()
  dataApoints = data["dataA_X"]
  dataAlabels = data["dataA_Y"]
  dataBpoints = data["dataB_X"]
  dataBlabels = data["dataB_Y"]
  dataCpoints = data["dataC_X"]
  dataClabels = data["dataC_Y"]
  
  plt.figure()
  do_the_clustering(dataApoints,1)
  do_the_clustering(dataBpoints,2)
  do_the_clustering(dataCpoints,3)
  plt.show()