#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
4 Cross-validation and variogram model selection
"""

import numpy as np
from scipy.stats import chi2

import kriging


def orthonormal_residuals(x, y, v, model_function, return_values=False):
    """
    Cross validation function based on orthonormal residues.
    reference: Peter K Kitanidis, Introduction to geostatistics:
               applications in hydrogeology, 1997, pp. 86--96
    Arguments:
        x,y,v : the data points
        model_function : a handle to the variogram model function

    Returns:
        success : true if model is acceptable, false otherwise
        Q1: the mean of the normalized error
        Q2: the mean of the squared normalized error
            at locations xi,yi
        MSE : the mean square error
        cR : the geometric mean of the kriging variances
        v_est : estimated kriging values
        v_var : variances of estimated kriging values
        v : true values
    """
    n = x.shape[0]
    p = np.random.permutation(n)
    # random permutation of input
    x = x[p]
    y = y[p]
    v = v[p]
    # number of degrees of freedom
    df = n - 1

    kriging_error = np.zeros(df)
    squared_error = np.zeros(df)
    normalized_error = np.zeros(df)
    squared_norm_error = np.zeros(df)
    v_est = np.zeros(df)
    v_var = np.zeros(df)

    for i in np.arange(df):
        v_est[i], v_var[i] = kriging.ordinary(
            x[0:i + 1], y[0:i + 1], v[0:i + 1],
            x[i + 1], y[i + 1], model_function)
        kriging_error[i] = v_est[i] - v[i + 1]

    squared_error = kriging_error**2
    normalized_error = kriging_error / np.sqrt(v_var)
    squared_norm_error = normalized_error**2
    Q1 = np.mean(normalized_error)
    Q2 = np.mean(squared_norm_error)
    MSE = np.mean(squared_error)
    cR = np.exp(np.sum(np.log(v_var)) / df)
    L = chi2.ppf(0.025, df) / df
    U = chi2.ppf(0.975, df) / df
    success = (np.abs(Q1) <= 2 / np.sqrt(df)) and (Q2 <= U and Q2 >= L)
    if (return_values):
        return success, Q1, Q2, MSE, cR, v_est, v_var, v, normalized_error
    else:
        return success, Q1, Q2, MSE, cR
