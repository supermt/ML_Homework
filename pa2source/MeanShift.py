#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import random
import data_reading
import matplotlib.pyplot as plt


STOP_THRESHOLD = 1e-4
CLUSTER_THRESHOLD = 1e-1
MAX_STPES = 100

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)

class MeanShift(object):
    def __init__(self, kernel=gaussian_kernel):
        self.kernel = kernel

    def fit(self, points, kernel_bandwidth):
        print "Start fitting"
        shift_points = np.array(points)
        shifting = [True] * points.shape[0]
        steps = 0
        while True:
            max_dist = 0
            for i in range(0, len(shift_points)):
                if not shifting[i]:
                    continue
                p_shift_init = shift_points[i].copy()
                shift_points[i] = self._shift_point(shift_points[i], points, kernel_bandwidth)
                dist = distance(shift_points[i], p_shift_init)
                max_dist = max(max_dist, dist)
                shifting[i] = dist > STOP_THRESHOLD
            steps = steps + 1
            if(max_dist < STOP_THRESHOLD):
                break
            if (steps >= MAX_STPES):
                break
            print "step " + str(steps)
        cluster_ids = self._cluster_points(shift_points.tolist())
        return steps, shift_points, cluster_ids

    def _shift_point(self, point, points, kernel_bandwidth):
        shift_dim = np.zeros(len(point))
        scale = 0.0
        for p in points:
            dist = distance(point, p)
            weight = self.kernel(dist, kernel_bandwidth)
            shift_dim = shift_dim + p * weight
            scale += weight
        shift_dim = shift_dim / scale
        return shift_dim

    def _cluster_points(self, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if(len(cluster_ids) == 0):
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for center in cluster_centers:
                    dist = distance(point, center)
                    if(dist < CLUSTER_THRESHOLD):
                        cluster_ids.append(cluster_centers.index(center))
                if(len(cluster_ids) < i + 1):
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        return cluster_ids

def clustering(datapoints,bandwidth):
    points = datapoints.T
    matY = np.matrix(points, copy=True)
    mean_shifter = MeanShift()
    steps, shift_points, label = mean_shifter.fit(points, kernel_bandwidth=bandwidth)
    label = np.array(label)
    return label



def do_the_clustering(datapoints,n):
  points = datapoints.T
  matY = np.matrix(points, copy=True)
  mean_shifter = MeanShift()
  h = 1.9
  steps, shift_points, label = mean_shifter.fit(points, kernel_bandwidth=h)
  plt.subplot(2,2,n)
  label = np.array(label)
  for i in set(label):
    plt.scatter((points[label == i])[:,0],(points[label == i])[:,1])
  print "produce " + str(i+1) + " clusters using kernel_bandwidth " + str(h) + " with in " + str(steps) + " stpes"

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