#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
Exploratory data analysis
Exercise 1
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    Plots position and concentration of Cobalt (mg/kg) from the data.
    """
    jura_data = np.genfromtxt('data.txt', names=True)
    plt.scatter(jura_data['X'], jura_data['Y'], s=30, c=jura_data['Co'])
    plt.colorbar()
    plt.axis('scaled')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.title('Map of Cobalt measurements (mg/kg)')
    plt.show()


if __name__ == '__main__':
    main()
