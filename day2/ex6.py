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
    hc, gc = variogram.cloud(jura_data['X'], jura_data['Y'], y)
    nlag = 50
    hmax = 2*np.max(hc)/3
    lag = hmax/nlag
    he, ge = variogram.experimental(hc, gc, lag, nlag)

    hm = np.linspace(0, hmax, 300)
    vm = model_function(hm)
    cm = covmodel(hm)

    plt.figure(1)
    plt.scatter(he, ge)
    plt.plot(hm, vm)
    plt.plot(hm, cm)
    plt.xlabel('h')
    plt.ylabel('gamma(h)')

    nx = 60
    ny = 60
    xmin = 1.6
    xmax = 4.2
    ymin = 0.8
    ymax = 3.4

    xi, yi = np.meshgrid( np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny) )
    v_est, v_var = kriging.ordinary_mesh(jura_data['X'], jura_data['Y'], y, xi.flatten(), yi.flatten(), model_function)
    v_sim1 = sgs.conditional(jura_data['X'], jura_data['Y'], y, xi.flatten(), yi.flatten(), covmodel, 15)
    v_sim2 = sgs.conditional(jura_data['X'], jura_data['Y'], y, xi.flatten(), yi.flatten(), covmodel, 15)
    v_sim3 = sgs.conditional(jura_data['X'], jura_data['Y'], y, xi.flatten(), yi.flatten(), covmodel, 15)

    plt.figure(2)
    plt.subplot(221)
    grid = v_est.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Kriging')
    plt.colorbar()
    
    plt.subplot(222)
    grid = v_sim1.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Simulation 1')
    plt.colorbar()
    
    plt.subplot(223)
    grid = v_sim2.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Simulation 2')
    plt.colorbar()
    
    plt.subplot(224)
    grid = v_sim3.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Simulation 3')
    plt.colorbar()

    real_est = ns.back(v_est)
    real_sim1 = ns.back(v_sim1)
    real_sim2 = ns.back(v_sim2)
    real_sim3 = ns.back(v_sim3)
    
    plt.figure(3)
    plt.subplot(221)
    grid = real_est.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Kriging')
    plt.colorbar()
    
    plt.subplot(222)
    grid = real_sim1.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Simulation 1')
    plt.colorbar()
    
    plt.subplot(223)
    grid = real_sim2.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Simulation 2')
    plt.colorbar()
    
    plt.subplot(224)
    grid = real_sim3.reshape((nx,ny))
    plt.imshow(grid, extent=(xi.min(),xi.max(),yi.min(),yi.max()), interpolation='nearest')
    plt.title('Simulation 3')
    plt.colorbar()

    # histograms
    plt.figure(4)
    plt.subplot(221)
    plt.hist(jura_data['Pb'])
    plt.title('Pb measurements')
    
    plt.subplot(222)
    plt.hist(real_est)
    plt.title('Kriged values')
    
    plt.subplot(223)
    plt.hist(real_sim1)
    plt.title('Simulated 1 values')
    
    plt.subplot(224)
    plt.hist(real_sim2)
    plt.title('Simulated 2 values')
    
    plt.show()

def model_function(h):
    return model.exponential(h, 0.65, 1.5) + model.nugget(h, 0.35)

def covmodel(h):
    return 1 - model_function(h)

if __name__ == '__main__':
    main()
