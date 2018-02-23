#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
2 Variogram modeling
Exercise 6.
"""

import numpy as np
import matplotlib.pyplot as plt
import variogram
import model


def main():
    """
    Plots different variogram models and compares them to experimental data.
    """
    lag = 0.1
    nlag = 40
    jura_data = np.genfromtxt('data.txt', names=True)
    hc, gc = variogram.cloud(jura_data['X'], jura_data['Y'], jura_data['Co'])
    he, ge = variogram.experimental(hc, gc, lag, nlag)

    # Print one figure with 4 subplots, subplot syntax: (rows, columns, number)
    # Each figure represents a different variogram model for the same data
    plt.figure(1)

    # Gaussian
    plt.subplot(221)
    x = np.linspace(0, nlag * lag, 1000)
    y = model.gaussian(x, 10, 0.9) + 3.0
    plt.scatter(he, ge, s=30)
    variance = np.var(jura_data['Co'])
    plt.axhline(variance, linestyle='--')
    plt.xlim(xmin=0, xmax=lag * nlag)
    plt.ylim(ymin=0)
    plt.title('Gaussian')
    plt.plot(x, y)

    # Exponential
    plt.subplot(222)
    x = np.linspace(0, nlag * lag, 1000)
    y = model.exponential(x, 12.5, 1.5)
    plt.scatter(he, ge, s=30)
    variance = np.var(jura_data['Co'])
    plt.axhline(variance, linestyle='--')
    plt.xlim(xmin=0, xmax=lag * nlag)
    plt.ylim(ymin=0)
    plt.title('Exponential')
    plt.plot(x, y)

    # Spherical
    plt.subplot(223)
    x = np.linspace(0, nlag * lag, 1000)
    y = model.spherical(x, 13, 1.5)
    plt.scatter(he, ge, s=30)
    variance = np.var(jura_data['Co'])
    plt.axhline(variance, linestyle='--')
    plt.xlim(xmin=0, xmax=lag * nlag)
    plt.ylim(ymin=0)
    plt.xlabel('distance')
    plt.ylabel('gamma')
    plt.title('Spherical')
    plt.plot(x, y)

    # Pure nugget
    plt.subplot(224)
    x = np.linspace(0, nlag * lag, 1000)
    y = model.nugget(x, 13)
    plt.scatter(he, ge, s=30)  # marker size 's' set to 30
    variance = np.var(jura_data['Co'])
    plt.axhline(variance, linestyle='--')
    plt.xlim(xmin=0, xmax=lag * nlag)
    plt.ylim(ymin=0)
    plt.title('Nugget')
    plt.plot(x, y)

    plt.show()


if __name__ == '__main__':
    main()
