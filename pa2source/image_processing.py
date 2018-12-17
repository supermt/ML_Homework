#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pa2
import numpy as np
import pylab as pl
import PIL.Image as Image
import scipy.io as sio
import Kmeans
# import EMGMM
from sklearn.mixture import GaussianMixture



from sklearn.cluster import MeanShift

def demo():
    import scipy.cluster.vq as vq

    ## load and show image
    img = Image.open('images/117054.jpg')
    K=2
    pl.figure(str(K))
    pl.subplot(2,3,1)
    pl.imshow(img)
    pl.subplot(2,3,4)
    pl.imshow(img)

    ## extract features from image (step size = 7)
    X,L = pa2.getfeatures(img, 7)

    X = vq.whiten(X.T)
    print "Start Learning"
    C,Y = vq.kmeans2(X, K, iter=1000, minit='random')

    Y = Y + 1
    # make segmentation image from labels
    segm = pa2.labels2seg(Y,L)
    pl.subplot(2,3,2)
    pl.imshow(segm)
    
    # color the segmentation image
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(2,3,3)
    pl.imshow(csegm)
    Y = GaussianMixture(n_components=K, covariance_type='full').fit(X)
    Y = Y.predict(X) + 1
    # Y = EMGMM.clustering(X, K)
    # make segmentation image from labels
    segm = pa2.labels2seg(Y,L)
    pl.subplot(2,3,5)
    pl.imshow(segm)
    
    # color the segmentation image
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(2,3,6)
    pl.imshow(csegm)


    # # Y = MeanShift.clustering((vq.whiten(X.T)).T, 5)
    # Y = MeanShift(bandwidth=0.9).fit((vq.whiten(X.T)))
    # Y = Y.labels_ + 1
    # # make segmentation image from labels
    # segm = pa2.labels2seg(Y,L)
    # pl.subplot(3,3,8)
    # pl.imshow(segm)
    
    # # color the segmentation image
    # csegm = pa2.colorsegms(segm, img)
    # pl.subplot(3,3,9)
    # pl.imshow(csegm)
    pl.show()

def main():
    demo()
if __name__ == '__main__':
    main()
