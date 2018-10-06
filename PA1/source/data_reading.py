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
  count_data , count_keys= readMatFile("count_data.mat")
  poly_data , poly_keys = readMatFile("poly_data.mat")
  print (poly_keys)
  
if __name__ == '__main__':
  main()