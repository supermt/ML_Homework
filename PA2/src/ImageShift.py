#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pa2
import numpy as np
import pylab as pl
import PIL.Image as Image
import scipy.io as sio
from sklearn.cluster import MeanShift

def demo():
    import scipy.cluster.vq as vq

    ## load and show image
    img = Image.open('images/12003.jpg')
    pl.subplot(3,3,1)
    pl.imshow(img)
    pl.subplot(3,3,4)
    pl.imshow(img)
    pl.subplot(3,3,7)
    pl.imshow(img)

    ## extract features from image (step size = 7)
    X,L = pa2.getfeatures(img, 7)

    X = vq.whiten(X.T)

    for i in range(3):
      Y = MeanShift(bandwidth=0.9*(i+1)).fit(X)
      Y = Y.labels_ + 1 # Use matlab 1-index labeling
      # make segmentation image from labels
      segm = pa2.labels2seg(Y,L)
      pl.subplot(3,3,(3 * i + 2))
      pl.imshow(segm)
      
      # color the segmentation image
      csegm = pa2.colorsegms(segm, img)
      pl.subplot(3,3,(3 * i + 3))
      pl.imshow(csegm)
    
    pl.show()

def main():
    demo()
if __name__ == '__main__':
    main()
