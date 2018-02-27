#!/usr/bin/env python3

"""
13.02.2018
Part 2 - Estimations versus simulations
1 Uniform random number generator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

def congruential(n, seed, m, a, c):
    """
    Congruential uniform pseudo random number generator
    Input:
      n: number of random numbers to generate
      seed: initial seed
      m: parameter of PRNG
      a: parameter of PRNG
      c: parameter of PRNG

    Returns:
      u : vector of random numbers
    """
    assert(n > 0)
    x = np.zeros(n)
    x[0] = seed%m
    for i in np.arange(1, n):
        x[i] = (a*x[i-1]+c) % m
    return x/m

def gaussian(n, seed):
    """
    Gaussian random number generator. mean = 0, variance = 1
    Input:
        n: size of random vector to be returned
        seed: seed for the generator

    Returns:
        y : vector of random gaussian numbers
    """
    m = 2**31 -1
    a = 48271
    c = 0
    u = congruential(n, seed, m, a, c)
    y = np.sqrt(2) * erfinv(2*u-1)
    return y

