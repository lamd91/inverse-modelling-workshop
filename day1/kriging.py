import numpy as np
from scipy.spatial import distance
from scipy.linalg import lu_factor, lu_solve

"""
12.02.2018
Exploratory data analysis.
Part 1 - Variograms and kriging
3 Kriging
"""

def ordinary(x,y,v,xi,yi,model_function):
    """
    input:
      x,y,v : the data points
      xi,yi : the point where a kriging interpolation is requested
      variogram : a handle to the variogram model function
    model_function:
      v_est,v_var : the estimated values and the kriging variance at location (xi,yi)
    """
    G = _G_matrix(x, y, model_function)
    g = _g_vector(x, y, xi, yi, model_function)
    lambda_vec = np.linalg.solve(G,g)
    v_est = np.sum(lambda_vec[0:-1]*v)
    v_var = np.sum(-lambda_vec*g)
    return v_est, v_var

def ordinary_mesh(x, y, v, xi, yi, model_function):
    """
        
    """
    G = _G_matrix(x, y, model_function)
    lu, piv = lu_factor(G)
    nb_points = xi.shape[0]
    v_est = np.zeros(nb_points)
    v_var = np.zeros(nb_points)
    for i in np.arange(nb_points):
        g = _g_vector(x, y, xi[i], yi[i], model_function)
        lambda_vec = lu_solve((lu, piv),g)
        v_est[i] = np.sum(lambda_vec[0:-1]*v)
        v_var[i] = np.sum(-lambda_vec*g)
    return v_est, v_var

def _G_matrix(x, y, model_function):
    n = x.shape[0]
    X = np.hstack((x[:,np.newaxis], y[:,np.newaxis]))
    G = np.ones((n+1,n+1))
    G[0:-1,0:-1] = -model_function(distance.squareform(distance.pdist(X)))
    G[-1,-1] = 0
    return G

def _g_vector(x, y, xi, yi, model_function):
    n = x.shape[0]
    g = np.ones(n+1)
    g[0:-1] = -model_function(np.sqrt((xi-x)**2 + (yi-y)**2))
    return g
