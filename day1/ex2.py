#!/usr/bin/env python3

"""
12.02.2018
Part 1 - Variograms and kriging
1 Exploratory data analysis
Exercise 2.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    Evaluates basic statistics of Cobalt concentration (mg/kg)
    """
    jura_data = np.genfromtxt('data.txt', names=True)
    print('Cobalt concentration (mg/kg) basic statistics')
    print('Number of points:', np.size(jura_data['Co']))
    print('Minimum:', np.min(jura_data['Co']))
    print('Mean', np.mean(jura_data['Co']))
    print('Maximum:', np.max(jura_data['Co']))
    print('Variance:', np.var(jura_data['Co']))
    print('Standard deviation:', np.sqrt(np.var(jura_data['Co'])))


if __name__ == '__main__':
    main()
