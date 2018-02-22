#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
5 Mapping the cobalt concentrations in soil
Exercise 11.
"""

import numpy as np
import matplotlib.pyplot as plt
import model
import kriging


def model_function(h):
    """
    Model function used in the exercise 11
    """
    return model.exponential(h, 7, 2) + model.nugget(h, 2)


def main():
    """
    Computes and prints cobalt concentration estimations and their variance
    on a grid.
    """
    nx = 50 # number of cells in x direction
    ny = 50 # number of cells in y direction
    xmax = 5.0 # range of x (km)
    ymax = 6.0 # range of y (km)
    x = np.linspace(0, xmax, nx)
    y = np.linspace(0, ymax, ny)
    xv, yv = np.meshgrid(x, y) # grid on which we apply kriging
    jura_data = np.genfromtxt('data.txt', names=True)   
    v_est, v_var = kriging.ordinary_mesh(
        jura_data['X'], jura_data['Y'], jura_data['Co'],
        xv.flatten(), yv.flatten(), model_function)

    # Plot map of estimates and map of variances
    plt.figure(1)
    
    plt.subplot(121)
    plt.pcolor(xv, yv, v_est.reshape(nx, ny))
    plt.colorbar()
    # plot also measurements points with 'x' markers
    plt.scatter(jura_data['X'], jura_data['Y'], marker='x', c='k')
    plt.axis('scaled')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.xlim(xmin=0, xmax=xmax)
    plt.ylim(ymin=0, ymax=ymax)
    plt.title('Kriging estimated values')

    plt.subplot(122)
    plt.pcolor(xv, yv, v_var.reshape(nx, ny))
    plt.colorbar()
    # plot also measurements points with 'x' markers
    plt.scatter(jura_data['X'], jura_data['Y'], marker='x', c='k')
    plt.axis('scaled')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.xlim(xmin=0, xmax=xmax)
    plt.ylim(ymin=0, ymax=ymax)
    plt.title('Kriging variances')

    plt.show()


if __name__ == '__main__':
    main()
