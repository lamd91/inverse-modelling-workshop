#!/usr/bin/env python3

"""
13.02.2018
Part 2 - Estimations versus simulations
3 Multi-Gaussian random field
"""

import numpy as np
import matplotlib.pyplot as plt

import sgs

def main():
    """
    Application of unconditional sequential gaussian simulation on a grid
    for exponential and gaussian covariance models
    """
    nx, ny = (60, 60)
    x = np.linspace(1, nx, nx)
    y = np.linspace(1, ny, ny)
    xv, yv = np.meshgrid(x, y)
    xi = xv.flatten()
    yi = yv.flatten()
    nmax = 20

    plt.figure(1)
    plt.subplot(221)
    vsim = sgs.unconditional(xi, yi, covmodel_exp, nmax)
    grid = vsim.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Exponential 1')
    
    plt.subplot(222)
    vsim = sgs.unconditional(xi, yi, covmodel_exp, nmax)
    grid = vsim.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Exponential 2')
    
    plt.subplot(223)
    vsim = sgs.unconditional(xi, yi, covmodel_gauss, nmax)
    grid = vsim.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Gaussian 1')
    
    plt.subplot(224)
    vsim = sgs.unconditional(xi, yi, covmodel_gauss, nmax)
    grid = vsim.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Gaussian 2')
    plt.show()

def covmodel_exp(x):
    sill = 1
    range = 10
    return sill*np.exp(-3*np.abs(x)/range)

def covmodel_gauss(x):
    sill = 1
    range = 10
    return sill*np.exp(-3*(np.abs(x)/range)**2)

if __name__ == '__main__':
    main()
