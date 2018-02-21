#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
2 Variogram modeling
Exercise 6.
"""

import numpy as np
import matplotlib.pyplot as plt
import variogram
import model

def main():
    lag = 0.1
    nlag = 40
    jura_data = np.genfromtxt('data.txt', names=True)
    hc, gc = variogram.cloud(jura_data['X'], jura_data['Y'], jura_data['Co'])
    he, ge = variogram.experimental(hc, gc, lag, nlag)
    
    plt.scatter(he, ge, s=30) # marker size 's' set to 30
    variance = np.var(jura_data['Co'])
    plt.axhline(variance, linestyle='--')

    plt.figure(1)
    plt.subplot(221)
    x = np.linspace(0, nlag*lag, 1000)
    y = model.gaussian(x, 10, 0.9) + 3.0
    plt.scatter(he, ge, s=30) # marker size 's' set to 30
    variance = np.var(jura_data['Co'])
    plt.axhline(variance, linestyle='--')
    plt.title('Gaussian')
    plt.plot(x,y)
    
    plt.subplot(222)
    x = np.linspace(0, nlag*lag, 1000)
    y = model.exponential(x, 12.5, 1.5)
    plt.scatter(he, ge, s=30) # marker size 's' set to 30
    variance = np.var(jura_data['Co'])
    plt.axhline(variance, linestyle='--')
    plt.title('Exponential')
    plt.plot(x,y)
    
    plt.subplot(223)
    x = np.linspace(0, nlag*lag, 1000)
    y = model.spherical(x, 13, 1.5)
    plt.scatter(he, ge, s=30) # marker size 's' set to 30
    variance = np.var(jura_data['Co'])
    plt.axhline(variance, linestyle='--')
    plt.title('Spherical')
    plt.plot(x,y)
    
    plt.subplot(224)
    x = np.linspace(0, nlag*lag, 1000)
    y = model.nugget(x, 13, 1.5)
    plt.scatter(he, ge, s=30) # marker size 's' set to 30
    variance = np.var(jura_data['Co'])
    plt.axhline(variance, linestyle='--')
    plt.title('Nugget')
    plt.plot(x,y)
    
    #x = np.linspace(0, nlag*lag, 1000)
    #y = vario_gaussian(x, 13, 1.5)
    #plt.plot(x,y)
    #plt.xlabel('distance')
    #plt.ylabel('$\gamma$')
    #plt.xlim(xmin=0, xmax=lag*nlag)
    #plt.ylim(ymin=0)
    
    plt.show()

if __name__ == '__main__':
    main()
