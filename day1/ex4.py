#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
1 Exploratory data analysis
Exercise 4.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import variogram

def main():
    lag = 0.1
    nlag = 40
    jura_data = np.genfromtxt('data.txt', names=True)
    hc, gc = variogram.cloud(jura_data['X'], jura_data['Y'], jura_data['Co'])
    he, ge = variogram.experimental(hc, gc, lag, nlag)
    
    plt.scatter(he, ge, s=30) # marker size 's' set to 30
    variance = np.var(jura_data['Co'])
    plt.axhline(variance, linestyle='--')
    plt.rc('text', usetex=True) # to obtain 'gamma' typeset by tex

    x = np.linspace(0, nlag*lag, 1000)
    y = vario_gaussian(x, 13, 1.5)
    plt.plot(x,y)
    plt.xlabel('distance')
    plt.ylabel('$\gamma$')
    plt.xlim(xmin=0, xmax=lag*nlag)
    plt.ylim(ymin=0)
    plt.title('Experimental Variogram of Cobalt measurments (mg/kg)')
    plt.show()

if __name__ == '__main__':
    main()
