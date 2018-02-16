#!/usr/bin/env python3

"""
12.02.2018
Exploratory data analysis. Part 1 - Variograms and kriging
Exercise 3
"""

import numpy as np
import matplotlib.pyplot as plt
import variograms as vrg

def main():
    jura_data = np.genfromtxt('data.txt', names=True)
    hc, gc = vrg.variogram_cloud(jura_data['X'], jura_data['Y'], jura_data['Co'])
    plt.scatter(hc, gc, s=10)
    plt.xlabel('X')
    plt.ylabel('gamma')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.title('Variogram of Cobalt measurments (mg/kg)')
    plt.show()


if __name__ == '__main__':
    main()
