#!/usr/bin/env python3

import numpy as np
import kriging
from scipy.stats import chi2

"""
12.02.2018
Part 1 - Variograms and kriging
2 Variogram modelling
"""

def orthonormal_residuals(x, y, v, model_function):
    """
% input: 
%  x,y,v : the data points 
%  variogram : a handle to the variogram model function
%  verbos : the function prints and display its results graphically
% output:
%  success : true if model is acceptable, false otherwise
%  Q1: the mean of the normalized error
%  Q2: the mean of the squared normalized error
%           at locations xi,yi
%  MSE : the mean square error 
%  cR : the geometric mean of the kriging variances
%  verbose: if true the function prints the results and plot the graphs
    """
    n = x.shape[0]
    p = np.random.permutation(n)
    x = x[p]
    y = y[p]
    v = v[p]
    df = n-1

    kriging_error = np.zeros(df)
    squared_error = np.zeros(df)
    normalized_error = np.zeros(df)
    squared_norm_error = np.zeros(df)
    v_est = np.zeros(df)
    v_var = np.zeros(df)

    for i in np.arange(df):
        v_est[i], v_var[i] = kriging.ordinary(x[0:i+1], y[0:i+1], v[0:i+1], x[i+1], y[i+1], model_function)
        kriging_error[i] = v_est[i] - v[i+1]
        if kriging_error[i] > 100:
            print(x[i+1], y[i+1], v[i+1], v_est[i])
        squared_error[i] = kriging_error[i]**2
        normalized_error[i] = kriging_error[i]/np.sqrt(v_var[i])
        squared_norm_error[i] = normalized_error[i]**2
    Q1 = np.sum(normalized_error)/df
    Q2 = np.sum(squared_norm_error)/df
    MSE = np.sum(squared_error)/df
    cR = np.exp( np.sum( np.log(v_var))/df)
    L = chi2.ppf(0.025, df) / df
    U = chi2.ppf(0.975, df) / df
    success = (np.abs(Q1)<= 2/np.sqrt(df)) and (Q2<=U and Q2>=L)
    return success, Q1, Q2, MSE, cR
