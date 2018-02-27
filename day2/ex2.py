#!/usr/bin/env python3

"""
13.02.2018
Part 2 - Estimations versus simulations
2 Gaussian random number generator
Exercise 2.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import prng


def main():
    """
    Plots histogram and cross plot of gaussian random numbers.
    """
    mu = 0
    sigma = 1
    seed = 1231
    n = 100000
    u = prng.gaussian(n, seed)
    plt.figure(1)
    plt.subplot(121)
    plt.title('Histogram')
    plt.hist(u, normed=1)
    x_draw = np.linspace(mu - 5*sigma, mu + 5*sigma, n)
    plt.plot(x_draw, norm.pdf(x_draw, mu, sigma))  

    plt.subplot(122)
    plt.title('Cross plot of gaussian PRNG')
    plt.scatter(u[0:-1], u[1:], s=0.1)
    plt.show()


if __name__ == '__main__':
    main()
