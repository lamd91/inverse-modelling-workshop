#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
2 Variogram modelling

Provides variogram models.
"""

import numpy as np


def gaussian(h, sill, range):
    """
    Gaussian variogram model
    """
    return sill * (1 - np.exp(-3 * (h / range)**2))


def exponential(h, sill, range):
    """
    Exponential variogram model
    """
    return sill * (1 - np.exp(-3 * np.abs(h) / range))


def sinus_cardinal(h, sill, range):
    """
    Sinus cardinal variogram model
    """
    return sill * (1 - (range / h) * np.sin(h / range))


def hyperbolic(h, sill, range):
    """
    Hyperbolic variogram model
    """
    return sill / (1 + h / range)


def nugget(h, nugget):
    """
    Pure nugget variogram model
    """
    return (h == 0) * 0 + (h > 0) * nugget


def spherical(h, sill, range):
    """
    Spherical variogram model
    """
    return sill * (3 * h / (2 * range) - 0.5 * (h / range) **
                   3) * (h < range) + sill * (h >= range)


def linear(h, sill, range):
    """
    Linear variogram model
    """
    return sill * h / range


def stable(h, sill, range):
    """
    Stable variogram model
    """
    return sill * (1 - np.exp(-3 * np.sqrt(h / range)))
