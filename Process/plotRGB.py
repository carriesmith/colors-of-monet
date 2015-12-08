import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn

def plotRGB(pixels, centroids = None, labels = None):
    
    pixs = pixels.copy()
    
    # If passed labels, points will have borders
    if (isinstance(labels, np.ndarray)):
        pixs['labels'] = labels
        lw = 1
    else:
        pixs['labels'] = 0
        lw = 0
    
    # Initialize single figure
    plt.figure(1, figsize = (16,4)) 
    
    # Border colours
    border = 'krb'

    for pixel in pixs.iterrows():
        plt.subplot(131) 
        col = [(pixel[1][0]/255.0, pixel[1][1]/255.0, pixel[1][2]/255.0)]
        rg = plt.scatter(pixel[1][0], pixel[1][1], c=col, linewidths=lw, s=100, alpha=.6, edgecolor=border[pixel[1][3]])

        plt.subplot(132) 
        gb = plt.scatter(pixel[1][1], pixel[1][2], c=col, linewidths=lw, s=100, alpha=.6, edgecolor=border[pixel[1][3]])

        plt.subplot(133) 
        br = plt.scatter(pixel[1][2], pixel[1][0], c=col, linewidths=lw, s=100, alpha=.6, edgecolor=border[pixel[1][3]])

    # If centroids are given they are added to plots. White dot with a coloured dot on top.
    if (isinstance(centroids, np.ndarray)):
        plt.subplot(131)
        rg = plt.scatter(centroids[0,0], centroids[0,1], c='w',linewidths=0, s=40, alpha=1)
        rg = plt.scatter(centroids[0,0], centroids[0,1], c='g',linewidths=0, s=10, alpha=1)
        rg = plt.scatter(centroids[1,0], centroids[1,1], c='w',linewidths=0, s=40, alpha=1)
        rg = plt.scatter(centroids[1,0], centroids[1,1], c='r',linewidths=0, s=10, alpha=1)
        rg = plt.scatter(centroids[2,0], centroids[2,1], c='w',linewidths=0, s=40, alpha=1)
        rg = plt.scatter(centroids[2,0], centroids[2,1], c='b',linewidths=0, s=10, alpha=1)

        plt.subplot(132) 
        rg = plt.scatter(centroids[0,1], centroids[0,2], c='w',linewidths=0, s=40, alpha=1)
        rg = plt.scatter(centroids[0,1], centroids[0,2], c='g',linewidths=0, s=10, alpha=1)
        rg = plt.scatter(centroids[1,1], centroids[1,2], c='w',linewidths=0, s=40, alpha=1)
        rg = plt.scatter(centroids[1,1], centroids[1,2], c='r',linewidths=0, s=10, alpha=1)
        rg = plt.scatter(centroids[2,1], centroids[2,2], c='w',linewidths=0, s=40, alpha=1)
        rg = plt.scatter(centroids[2,1], centroids[2,2], c='b',linewidths=0, s=10, alpha=1)

        plt.subplot(133) 
        rg = plt.scatter(centroids[0,2], centroids[0,0], c='w',linewidths=0, s=40, alpha=1)
        rg = plt.scatter(centroids[0,2], centroids[0,0], c='g',linewidths=0, s=10, alpha=1)
        rg = plt.scatter(centroids[1,2], centroids[1,0], c='w',linewidths=0, s=40, alpha=1)
        rg = plt.scatter(centroids[1,2], centroids[1,0], c='r',linewidths=0, s=10, alpha=1)
        rg = plt.scatter(centroids[2,2], centroids[2,0], c='w',linewidths=0, s=40, alpha=1)
        rg = plt.scatter(centroids[2,2], centroids[2,0], c='b',linewidths=0, s=10, alpha=1)

    # Label axes
    plt.subplot(131)     
    rg = plt.xlabel('r'), plt.ylabel('g')

    plt.subplot(132)     
    gb = plt.xlabel('g'), plt.ylabel('b')

    plt.subplot(133)     
    gb = plt.xlabel('b'), plt.ylabel('r')