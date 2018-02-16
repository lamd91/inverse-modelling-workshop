#!/usr/bin/env python3

"""
12.02.2018
Exploratory data analysis. Part 1 - Variograms and kriging
Exercise 9
"""

import numpy as np
import matplotlib.pyplot as plt
import variograms as vrg

def vario(h):
    return vrg.vexponential(h,7,2) + vrg.vnugget(h,2)

def main():
    t = np.loadtxt('data_py.txt')
    print(vrg.ordinary_kriging(t[:,0], t[:,1], t[:,4],3.0, 3.0,vario))
    
    
    
if __name__ == '__main__':
    main()


