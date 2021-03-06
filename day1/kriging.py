"""
12-13.02.2018
Part 1 - Variograms and kriging (ordinary kriging)
Part 2 - Estimations versus simulations (simple kriging)

Library of kriging functions
"""

import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import lu_factor, lu_solve, solve


def ordinary(x, y, v, xi, yi, model_function):
    """
    Ordinary kriging implementation.
    Arguments:
      x,y,v : the data points
      xi,yi : the point where a kriging interpolation is requested
      model_function: variogram model function
    Results:
      v_est : the estimated value at location (xi,yi)
      v_var : kriging variance at location (xi,yi)
    """
    G = _G_matrix(x, y, model_function)
    g = _g_vector(x, y, xi, yi, model_function)
    lambda_vec = np.linalg.solve(G, g)
    v_est = np.sum(lambda_vec[0:-1] * v)
    v_var = np.sum(-lambda_vec * g)
    return v_est, v_var


def ordinary_mesh(x, y, v, xi, yi, model_function):
    """
    Ordinary kriging implementation returning kriging values on a mesh
    Arguments:
      x,y,v : the data points
      xi,yi : the point where a kriging interpolation is requested
      model_function: variogram model function
    Results:
      v_est : array of estimated values at locations (xi,yi)
      v_var : array of kriging variances at locations (xi,yi)
    """
    G = _G_matrix(x, y, model_function)
    lu, piv = lu_factor(G)
    nb_points = xi.shape[0]
    v_est = np.zeros(nb_points)
    v_var = np.zeros(nb_points)
    for i in np.arange(nb_points):
        g = _g_vector(x, y, xi[i], yi[i], model_function)
        lambda_vec = lu_solve((lu, piv), g)
        v_est[i] = np.sum(lambda_vec[0:-1] * v)
        v_var[i] = np.sum(-lambda_vec * g)
    return v_est, v_var

def simple(x, y, v, xi, yi, covmodel, mu):
    """
    Simple kriging implementation
    Arguments:
        x, y, v: data points
        xi, yi: point where kriging interpolation is requested
        covmodel: covariance model function
        mu: mean
    Results:
        v_est : array of estimated values at locations (xi,yi)
        v_var : array of kriging variances at locations (xi,yi)
    """
    n = x.shape[0]
    if n==0:
        d = 0;
    else:
        X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
        d = squareform( pdist(X))
    C = covmodel(d);
    c = covmodel( np.sqrt( (xi-x)**2 + (yi-y)**2 ) )
    l = solve(C,c)
    
    v_est = np.sum(l*(v-mu)) + mu
    v_var = covmodel(0)-np.sum(l*c)
    return v_est, v_var

def _G_matrix(x, y, model_function):
    """
    Builds G kriging matrix
    """
    n = x.shape[0]
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    G = np.ones((n + 1, n + 1))
    G[0:-1, 0:-1] = -model_function(squareform(pdist(X)))
    G[-1, -1] = 0
    return G


def _g_vector(x, y, xi, yi, model_function):
    """
    Builds g kriging vector
    """
    n = x.shape[0]
    g = np.ones(n + 1)
    g[0:-1] = -model_function(np.sqrt((xi - x)**2 + (yi - y)**2))
    return g
