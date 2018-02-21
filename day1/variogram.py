#!/usr/bin/env python3

"""
12.02.2018
Exploratory data analysis.
Part 1 - Variograms and kriging
Functions for Exercises 3,4.
"""

import numpy as np
from scipy.spatial import distance

def cloud(x, y, v):
    """
    input:
      x,y : vectors containing the coordinates of the points
      v : vector containing the values of the measurements at those locations
    output:
      hc, gc : the points of the variogram cloud
      hc contains the lag values and gc the squared difference
    """
    X = np.hstack((x[:,np.newaxis], y[:,np.newaxis])) #reshape((-1,1))
    hc = distance.pdist(X)
    gc = 0.5 * distance.pdist(v[:,np.newaxis]) **2
    return (hc, gc)

def experimental(hc,gc,lag,nlag):
    """
    input:
      hc,gc : the points of the variogram cloud
      lag,nlag : the lag distance and number of lags (number of classes)
    output:
      he, ge : the points of the experimental variogram
      he contains the lag values and ge the experimental variogram
    """
    
    variogram_range = nlag*lag;
    he = np.linspace(lag/2, variogram_range - lag/2, nlag)
    gamma_cum = np.zeros(nlag)
    num_points = np.zeros(nlag)
    N = hc.shape[0]
    for i in np.arange(0,N):
        if (hc[i] < variogram_range):
            if hc[i] < 0:
                raise ValueError('Input must be a positive array')
            class_index = int((hc[i]/variogram_range) *nlag);
            num_points[class_index] += 1
            gamma_cum[class_index] += gc[i]
    ge = np.zeros(nlag)   
    for j in np.arange(0,nlag):
        if num_points[j] != 0:
            ge[j] = gamma_cum[j] / num_points[j]
        else:
            ge[j] = 0.0
    return he, ge
    

