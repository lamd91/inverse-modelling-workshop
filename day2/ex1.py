#!/usr/bin/env python3

"""
13.02.2018
Part 2 - Estimations versus simulations
1 Uniform random number generator
Exercise 1.
"""

import numpy as np
import matplotlib.pyplot as plt

import prng

def main():
    """
    Plots random numbers obtained using congruential PRNG.
    Compares results obtained with some parameters to Park and Miller standard.
    """
    n = 10000
    # Bad parameters
    seed = 1231
    m = 32768
    a = 327
    c = 111
    u_bad = prng.congruential(n, seed, m, a, c)

    # Park and Miller minimal standard
    seed = 1231
    m = 2**31-1
    a = 48271
    c = 0
    u_pm = prng.congruential(n, seed, m, a, c)

    plt.figure(1)
    plt.subplot(321)
    plt.scatter(np.arange(0,n),u_bad, s=0.1)
    plt.title('Series of PRN')

    plt.subplot(322)
    plt.scatter(np.arange(0,n),u_pm, s=0.1)
    plt.title('Series of PRN - Park and Miller')

    plt.subplot(323)
    plt.scatter(u_bad[0:-1], u_bad[1:], s=0.1)
    plt.title('Cross plot')

    plt.subplot(324)
    plt.scatter(u_pm[0:-1], u_pm[1:], s=0.1)
    plt.title('Cross plot - Park and Miller')

    plt.subplot(325)
    plt.hist(u_bad)
    plt.title('Distribution')

    plt.subplot(326)
    plt.hist(u_pm)
    plt.title('distribution - Park and Miller')
    
    plt.show()


if __name__ == '__main__':
    main()
