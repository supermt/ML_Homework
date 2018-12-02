#!/usr/bin/env python
#-*- coding:utf-8 -*-
import data_reading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

MAX_ITER=50

def distance(p1,p2):
  tmp = np.sum((p1-p2)**2)
  return np.sqrt(tmp)

def randomCenters(data,k):
  n = data.shape[0]
  centroids = np.zeros((k,n))
  for i in range(n):
    dmin, dmax = np.min(data[:,i]), np.max(data[:,i])
    centroids[:,i] = dmin + (dmax - dmin) * np.random.rand(k)
  return centroids

def should_stop(centers1, centers2):
  set1 = set([tuple(c) for c in centers1])
  set2 = set([tuple(c) for c in centers2])
  return (set1 == set2)

def Kmeans(data, k=4):
  np.seterr(divide='ignore',invalid='ignore')
  n = data.shape[1]
  centers = randomCenters(data,k)
  label = np.zeros(n,dtype=np.int)
  assement = np.zeros(n) 
  stop = False
  data = data.T

  iter_count = 0
  while not stop:
    old_centers = np.copy(centers)
    iter_count = iter_count+1
    for i in range(n):
      min_dist, min_index = np.inf, -1
      for j in range(k):
        dist = distance(data[i],centers[j])
        if dist < min_dist:
            min_dist, min_index = dist, j
            label[i] = j
      assement[i] = distance(data[i],centers[label[i]])**2
    for m in range(k):
      centers[m] = np.mean(data[label==m],axis=0)
    stop = should_stop(old_centers,centers)
    if iter_count >= MAX_ITER:
      stop=True
  return centers, label, np.sum(assement)


def do_the_clustering(datapoints,n):
  K = 4
  best_centers,best_label,best_assement = Kmeans(datapoints,K)
  for i in range(10):
    centers, label, assement = Kmeans(datapoints,K)
    if assement<=best_assement:
      best_assement=assement
      best_label = label   

  points = datapoints.T
  plt.subplot(2,2,n)
  for i in range(K):
    plt.scatter((points[best_label == i])[:,0],(points[best_label == i])[:,1])

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
