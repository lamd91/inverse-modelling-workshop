#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
2 Variogram modelling
"""

import numpy as np

def gaussian(h, sill, range):
    return sill*(1-np.exp(-3*(h/range)**2))

def exponential(h, sill, range):
    return sill*(1-np.exp(-3*np.abs(h)/range))

def sinus_cardinal(h, sill, range):
    return sill*(1-(range/h)*np.sin(h/range))

def hyperbolic(h, sill, range):
    return sill/(1+h/range)

def nugget(h, nugget):
    return (h==0)*0+(h>0)*nugget

def spherical(h, sill, range):
    return sill*(3*h/(2*range)-0.5*(h/range)**3)*(h < range) + sill*(h >= range)

def linear(h, sill, range):
    return sill*h/range

def stable(h, sill, range):
    return sill*(1-np.exp(-3*np.sqrt(h/range)))
