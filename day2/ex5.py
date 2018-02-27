#!/usr/bin/env python3

"""
13.02.2018
Part 2 - Estimations versus simulations
4 Normal score transform
Exercise 5.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import normal_score_transform as nscore


def main():
    """
    Application of normal score transform on Pb data.
    """
    jura_data = np.genfromtxt('data.txt', names=True)
    ns = nscore.transform(jura_data['Pb'])
    y = ns.direct(jura_data['Pb'])


    plt.figure(1)
    plt.subplot(221)
    plt.hist(jura_data['Pb'])
    plt.title('Pb (mg/kg)')

    plt.subplot(222)
    plt.hist(y)
    plt.title('Transformed variable Y')

    plt.subplot(223)
    plt.scatter(jura_data['Pb'], y)
    plt.title('Cross plot')

    plt.subplot(224)
    stats.probplot(y, plot=plt)
    plt.show()
    

if __name__ == '__main__':
    main()
