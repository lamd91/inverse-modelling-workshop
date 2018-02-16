#!/usr/bin/env python3

"""
12.02.2018
Exploratory data analysis. Part 1 - Variograms and kriging
Functions for Exercises 3,4.
"""

import numpy as np
import scipy.spatial

def variogram_cloud(x, y, v):
    """
    input:
      x,y : vectors containing the coordinates of the points
      v : vector containing the values of the measurements at those locations
    output:
      hc, gc : the points of the variogram cloud
      hc contains the lag values and gc the squared difference
    """
    X = np.hstack((x[:,np.newaxis], y[:,np.newaxis])) #reshape((-1,1))
    hc = scipy.spatial.distance.pdist(X)
    gc = 0.5 * scipy.spatial.distance.pdist(v[:,np.newaxis]) **2
    return (hc, gc)

def variogram_experimental(hc,gc,lag,nlag):
    """
    input:
      hc,gc : the points of the variogram cloud
      lag,nlag : the lag distance and number of lags (number of classes)
    output:
      he, ge : the points of the experimental variogram
      he contains the lag values and ge the experimental variogram
    """
    
    l = nlag*lag;
    he = np.linspace(lag/2, l - lag/2, nlag)
    N = hc.shape[0]
    gamma_cum = np.zeros(nlag)
    num_points = np.zeros(nlag)
    print(N)
    
    for i in np.arange(0,N):
        index = np.floor(nlag*hc[i]/l)
        if (index < nlag):
            num_points[index] = num_points[index] + 1
            gamma_cum[index] = gamma_cum[index]+ gc[i]
        
    ge = np.zeros(nlag)
    for j in np.arange(1,nlag):
        if num_points[j] != 0:
            ge[j] = gamma_cum[j] / num_points[j]
        else:
            ge[j] = 0.0
    return he, ge

def vario_gaussian(h, sill, range):
    return sill*(1 - np.exp(-3*(h/range)**2))
    
def vexponential(h, sill,range):
    return sill*(1 - np.exp(-3*np.abs(h/range)))
    
def vnugget(h, nugget):
    return(h==0)*0 + (h>0)*nugget
    
def ordinary_kriging(x,y,v,xi,yi,variogram):
    """
    input:
      x,y,v : the data points
      xi,yi : the point where a kriging interpolation is requested
      variogram : a handle to the variogram model function
    output:
      v_est,v_var : the estimated values and the kriging variance at location (xi,yi)
    """


    X = np.vstack((x, y)).T
    N = X.shape[0]
    X0 = np.array([xi,yi])
    #print(X)
    #print(X0)
    X_extended = np.vstack((X0,X))
    Dist = scipy.spatial.distance.pdist(X_extended)
    #print(Dist)
    g = -variogram(Dist[0:N+1])
    g[-1] = 1
    G = np.ones((N+1,N+1))
    G[0:-1,0:-1] = -variogram(scipy.spatial.distance.squareform(Dist[N:]))
    G[-1,-1] = 0
    #print(G)
    #print(g)
    l = np.linalg.solve(G,g)
    #print(G, g, l)
    v_est = np.sum(l[:-1]*v)
    v_var = np.sum(-l* g)
    return v_est, v_var

