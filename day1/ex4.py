#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
1 Exploratory data analysis
Exercise 4.
"""

import numpy as np
import matplotlib.pyplot as plt

import variogram
import model


def main():
    """
    Plots experimental variogram of cobalt concentration data.
    """
    lag = 0.1 # width of bin
    nlag = 40 # number of bins
    jura_data = np.genfromtxt('data.txt', names=True)
    hc, gc = variogram.cloud(jura_data['X'], jura_data['Y'], jura_data['Co'])
    he, ge = variogram.experimental(hc, gc, lag, nlag)

    plt.scatter(he, ge, s=30)  # marker size 's' set to 30
    variance = np.var(jura_data['Co'])
    # variance presented as dashed line
    plt.axhline(variance, linestyle='--') 
    plt.plot(x, y)
    # axes labels
    plt.xlabel('distance')
    plt.ylabel('gamma')
    # axes limits
    plt.xlim(xmin=0, xmax=lag * nlag)
    plt.ylim(ymin=0)
    plt.title('Experimental variogram of cobalt concentration (mg/kg)')
    plt.show()


if __name__ == '__main__':
    main()
