#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pa2
import numpy as np
import pylab as pl
import PIL.Image as Image
import scipy.io as sio

def demo():
    import scipy.cluster.vq as vq

    ## load and show image
    img = Image.open('images/12003.jpg')
    pl.subplot(1,3,1)
    pl.imshow(img)
    
    ## extract features from image (step size = 7)
    X,L = pa2.getfeatures(img, 7)
    print X

    ## Call kmeans function in scipy.  You need to write this yourself!
    C,Y = vq.kmeans2(vq.whiten(X.T), 2, iter=1000, minit='random')
    Y = Y + 1 # Use matlab 1-index labeling
    ## 

    # make segmentation image from labels
    segm = pa2.labels2seg(Y,L)
    pl.subplot(1,3,2)
    pl.imshow(segm)
    
    # color the segmentation image
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(1,3,3)
    pl.imshow(csegm)
    pl.show()

def main():
    demo()
if __name__ == '__main__':
    main()
