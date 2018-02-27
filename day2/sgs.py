"""
13.02.2018
Part 2 - Estimations versus simulations
"""

import numpy as np

import prng
# need to import module kriging from day 1
import sys
sys.path.insert(0, '../day1')
import kriging

def unconditional(xi, yi, covmodel, nmax):
    """
    Unconditional sequential gaussian simulation algorithm
    Input:
        xi, yi: grid points
        covmodel: covariance function
        nmax: maximum number of neighboring points for kriging
    Result:
        v_sim: simulated values at xi, yi
    """
    N = xi.shape[0]
    path = np.random.permutation(N)
    # rearrange vectors according to the random path
    xi = xi[path]
    yi = yi[path]
    v_sim = np.zeros(N)
    mu = 0
    seed = 1232
    randn = prng.gaussian(N, seed)
    std_dev = np.sqrt(covmodel(0))
    # gaussian random number of mean mu and standard deviation
    v_sim[0] = mu + randn[0] * std_dev
    for i in np.arange(1,N):
        if i > nmax:
            distances = (xi[0:i]-xi[i])**2 + (yi[0:i]-yi[i])**2
            # partition such that nmax smallest elements appear first
            order = np.argpartition(distances, nmax)
            # order vectors accordingly
            x_close = xi[order]
            y_close = yi[order]
            v_close = v_sim[order]
            # krig only with nmax closest points
            v_est, v_var = kriging.simple( x_close[0:nmax], y_close[0:nmax], v_close[0:nmax], xi[i], yi[i], covmodel, mu)
        else:
            v_est, v_var = kriging.simple( xi[0:i], yi[0:i], v_sim[0:i], xi[i], yi[i], covmodel, mu)
        v_sim[i] = randn[i]*np.sqrt(v_var) + v_est
    # permute the result back so that it matches input xi, yi
    v_sim = v_sim[np.argsort(path)]
    return v_sim

def conditional(x, y, v, xi, yi, covmodel, nmax):
    """
    Conditional sequential gaussian simulation algorithm
    Input:
        x, y: coordinates of known values
        v : values at coordinates x,y
        xi, yi: grid points
        covmodel: covariance function
        nmax: maximum number of neighboring points for kriging
    Result:
        v_sim: simulated values at xi, yi
    """
    N = xi.shape[0]
    n_cond = x.shape[0]
    path = np.random.permutation(N)
    # rearrange vectors according to the random path
    xi = xi[path]
    yi = yi[path]
    v_sim = np.zeros(N)
    mu = 0
    seed = 1232
    randn = prng.gaussian(N, seed)
    std_dev = np.sqrt(covmodel(0))
    # gaussian random number of mean mu and standard deviation
    v_sim[0] = mu + randn[0] * std_dev
    x_all = np.zeros(N+n_cond)
    y_all = np.zeros(N+n_cond)
    v_all = np.zeros(N+n_cond)
    x_all[0:n_cond] = x
    y_all[0:n_cond] = y
    v_all[0:n_cond] = v
    for i in np.arange(1,N):
        distances = (x_all[0:i+n_cond]-xi[i])**2 + (y_all[0:i+n_cond]-yi[i])**2
        # partition such that nmax smallest elements appear first
        order = np.argpartition(distances, nmax)
        # order vectors accordingly
        x_close = x_all[order]
        y_close = y_all[order]
        v_close = v_all[order]
        # krig only with nmax closest points
        v_est, v_var = kriging.simple( x_close[0:nmax], y_close[0:nmax], v_close[0:nmax], xi[i], yi[i], covmodel, mu)
        v_sim[i] = randn[i]*np.sqrt(v_var) + v_est
        v_all[i+n_cond] = v_sim[i]
    # permute the result back so that it matches input xi, yi
    v_sim = v_sim[np.argsort(path)]
    return v_sim
