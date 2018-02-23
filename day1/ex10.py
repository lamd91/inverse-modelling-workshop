#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
4 Cross-validation and variogram model selection
Exercise 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import model
import cross_validation
import variogram


def main():
    """
    Performs cross validation for different model variograms.
    """
    jura_data = np.genfromtxt('data.txt', names=True)
    x, y, v = jura_data['X'], jura_data['Y'], jura_data['Co']

    success, Q1, Q2, MSE, cR = cross_validation.orthonormal_residuals(
        x, y, v, model_function1)
    print('function type, success, Q1, Q2, MSE, cR')
    print('1', success, Q1, Q2, MSE, cR)
    success, Q1, Q2, MSE, cR = cross_validation.orthonormal_residuals(
        x, y, v, model_function2)
    print('2', success, Q1, Q2, MSE, cR)
    success, Q1, Q2, MSE, cR = cross_validation.orthonormal_residuals(
        x, y, v, model_function3)
    print('3', success, Q1, Q2, MSE, cR)
    success, Q1, Q2, MSE, cR = cross_validation.orthonormal_residuals(
        x, y, v, model_function4)
    print('4', success, Q1, Q2, MSE, cR)
    success, Q1, Q2, MSE, cR, v_est, v_var, v, normalized_error = cross_validation.orthonormal_residuals(
        x, y, v, model_function5, return_values=True)
    print('5', success, Q1, Q2, MSE, cR)

    plt.figure(1)

    plt.subplot(221)
    plt.scatter(v_est, v[1:], marker='o')
    plt.plot(v_est, v_est)
    plt.xlabel('Estimation')
    plt.ylabel('True value')
    plt.axis('scaled')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.subplot(222)
    plt.hist(normalized_error, normed=1)
    x_draw = np.linspace(-5, 5, 100)
    plt.plot(x_draw, norm.pdf(x_draw, 0, 1))
    plt.xlabel('Normalised error')
    plt.ylabel('pdf')
    plt.title('Histogram of normalised error')

    plt.subplot(223)
    plt.scatter(x[1:], y[1:], s=10 * np.abs(normalized_error),
                c=np.abs(normalized_error))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Normalised error')

    hc, gc = variogram.cloud(x[1:], y[1:, ], normalized_error)
    nlag = 8
    lag = (np.max((np.max(x), np.max(y))) -
           np.min((np.min(x), np.min(y)))) * 2 / 3 / nlag
    he, ge = variogram.experimental(hc, gc, lag, nlag)
    plt.subplot(224)
    plt.scatter(he, ge)
    plt.axhline(1, linestyle='--')
    plt.ylim([0.0, 1.1 * np.max(ge)])
    plt.xlabel('Estimation')
    plt.ylabel('True value')
    plt.title('Variogram of normalised error')

    plt.show()


def model_function1(h):
    """
    Model function 1
    """
    return model.exponential(h, 14.5, 1.7) + model.nugget(h, 0.5)


def model_function2(h):
    """
    Model function 2
    """
    return model.gaussian(h, 10.0, 0.9) + model.nugget(h, 3.0)


def model_function3(h):
    """
    Model function 3
    """
    return model.exponential(h, 12.0, 1.4) + model.nugget(h, 1.3)


def model_function4(h):
    """
    Model function 4
    """
    return model.spherical(h, 10.3, 1.1) + model.nugget(h, 2.5)


def model_function5(h):
    """
    Model function 5
    """
    return model.exponential(h, 12.5, 1.5) + model.nugget(h, 1.0)


if __name__ == '__main__':
    main()
