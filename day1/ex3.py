#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
1 Exploratory data analysis. 
Exercise 3.
"""

import numpy as np
import matplotlib.pyplot as plt
import variogram

def main():
    """
    Plots variogram cloud of Co concentration measurments.
    """
    jura_data = np.genfromtxt('data.txt', names=True)
    hc, gc = variogram.cloud(jura_data['X'], jura_data['Y'], jura_data['Co'])
    plt.scatter(hc, gc, s=10)
    plt.xlabel('distance')
    plt.ylabel('Squared differences of concentrations')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.title('Variogram cloud of Cobalt measurments (mg/kg)')
    plt.show()


if __name__ == '__main__':
    main()
