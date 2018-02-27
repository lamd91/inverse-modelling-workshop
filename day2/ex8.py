#!/usr/bin/env python3

"""
13.02.2018
Part 2 - Estimations versus simulations
5 Conditional simulations
Exercise 6.
Exercise 7.
"""

import numpy as np
import matplotlib.pyplot as plt

# need to import module kriging from day 1
import sys
sys.path.insert(0, '../day1')
import model
import variogram
import kriging
import sgs

import normal_score_transform as nscore


def main():
    """
    Application of conditional simulations to Pb data
    """
    jura_data = np.genfromtxt('data.txt', names=True)
    ns = nscore.transform(jura_data['Pb'])
    y = ns.direct(jura_data['Pb'])

    nx = 60
    ny = 60
    xmin = 1.6
    xmax = 4.2
    ymin = 0.8
    ymax = 3.4

    xi, yi = np.meshgrid( np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny) )
    v_est, v_var = kriging.ordinary_mesh(jura_data['X'], jura_data['Y'], y, xi.flatten(), yi.flatten(), model_function)

    nsim = 400
    v_sim = np.zeros((nx*ny, nsim))
    for i in np.arange(nsim):
        v_sim[:,i] = sgs.conditional(jura_data['X'], jura_data['Y'], y, xi.flatten(), yi.flatten(), covmodel, 15)
    v_mean = np.mean(v_sim, axis=1)
    v_var_sim = np.var(v_sim, axis=1)

    plt.figure(1)

    plt.subplot(221)
    grid = v_est.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Kriging')
    plt.colorbar()
    
    plt.subplot(222)
    grid = v_mean.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Simulation mean')
    plt.colorbar()

    plt.subplot(223)
    grid = v_var.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Kriging')
    plt.colorbar()
    
    plt.subplot(224)
    grid = v_var_sim.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Simulation variance')
    plt.colorbar()

    plt.show()

def model_function(h):
    return model.exponential(h, 0.65, 1.5) + model.nugget(h, 0.35)

def covmodel(h):
    return 1 - model_function(h)

if __name__ == '__main__':
    main()
