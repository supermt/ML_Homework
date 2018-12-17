#!/usr/bin/env python
#-*- coding:utf-8 -*-
import csv
import numpy as np

writer = csv.writer(open("test.csv","w"))
with open('hungarian.data', 'rb') as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
  cleaned = []
  for row in spamreader:
    if '?' not in row:
      cleaned.append(row)
      writer.writerow(row)