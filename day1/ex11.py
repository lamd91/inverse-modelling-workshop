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
    return model.exponential(h,7,2) + model.nugget(h,2)

def main():
    jura_data = np.genfromtxt('data.txt', names=True)
    nx = 50
    ny = 50
    x = np.linspace(0, 5.0, nx)
    y = np.linspace(0, 6.0, ny)
    xv, yv = np.meshgrid(x, y)
    v_est, v_var = kriging.ordinary_mesh(jura_data['X'], jura_data['Y'], jura_data['Co'], xv.flatten(), yv.flatten(), model_function)
    plt.figure(1)
    plt.subplot(121)
    plt.pcolor(xv, yv, v_est.reshape(nx, ny))
    plt.colorbar()
    plt.scatter(jura_data['X'], jura_data['Y'], marker='x', c='k')
    plt.title('Kriging estimated values')
    
    plt.subplot(122)
    plt.pcolor(xv, yv, v_var.reshape(nx, ny))
    plt.colorbar()
    plt.scatter(jura_data['X'], jura_data['Y'], marker='x', c='k')
    plt.title('Kriging variances')
    
    plt.show()
    
    
    
    
if __name__ == '__main__':
    main()


