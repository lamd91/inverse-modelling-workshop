#!/usr/bin/env python3

"""
12.02.2018
Exploratory data analysis. Part 1 - Variograms and kriging
Exercise 4
"""

import numpy as np
import matplotlib.pyplot as plt
import variograms as vrg

def main():
    t = np.loadtxt('data_py.txt')
    hc, gc = vrg.variogram_cloud(t[:,0], t[:,1], t[:,4])
    lag = 0.5
    nlag = 10
    he, ge = vrg.variogram_experimental(hc, gc, lag, nlag);
    plt.scatter(he, ge, s=30)
    plt.xlabel('distance')
    plt.ylabel('gamma')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.title('Experimental Variogram of Cobalt measurments (mg/kg)')
    plt.show()
    
    
    
if __name__ == '__main__':
    main()

"""
t = readtable('data.txt');
[hc, gc] = variogram_cloud(t.X, t.Y, t.Co);
lag = 0.25
nlag = 24
[he, ge] = variogram_experimental(hc, gc, lag, nlag);
figure(1)
scatter( he, ge, 'filled')
xlabel('distance'); ylabel('\gamma')
title('Variogram of Cobalt measurments mg/kg')
grid on
figure(2)
derivative = diff(ge)/lag;
plot(he(1:end-1), derivative)
"""
