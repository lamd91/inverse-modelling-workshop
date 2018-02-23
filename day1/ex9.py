#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
3 Kriging
Exercise 9.
"""

import numpy as np
import matplotlib.pyplot as plt

import model
import kriging


def model_function(h):
    """
    Model function used in the exercise 9.
    """
    sill = 12.5
    range = 1.5
    nugget = 1
    return model.exponential(h, sill, range) + nugget


def main():
    """
    Estimates the cobalt concentration at (3.0, 3.0) and compute variance.
    Provides also 95% confidence interval
    """
    xi = 3.0
    yi = 3.0
    jura_data = np.genfromtxt('data.txt', names=True)
    v_est, v_var = kriging.ordinary(
        jura_data['X'], jura_data['Y'], jura_data['Co'],
        xi, yi, model_function)
    print('Cobalt concentration:', v_est)
    print('Variance:', v_var)
    print(
        '95% confidence interval:',
        v_est - 2 * np.sqrt(v_var),
        v_est + 2 * np.sqrt(v_var))


if __name__ == '__main__':
    main()
