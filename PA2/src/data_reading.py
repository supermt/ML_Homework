#!/usr/bin/env python
#-*- coding:utf-8 -*-
import scipy.io as scio

def readMatFile(filename):
  datasets = scio.loadmat(filename);
  datakeys = []
  for data_key in datasets:
    if (data_key[0] != '_'):
      datakeys.append(data_key)

  return datasets , datakeys


def main():
  count_data , count_keys= readMatFile("cluster_data.mat")
  return count_data
  
if __name__ == '__main__':
  main()