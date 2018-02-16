#!/usr/bin/env python3

"""
12.02.2018
Exploratory data analysis. Part 1 - Variograms and kriging
Exercise 5
"""

import numpy as np
import matplotlib.pyplot as plt
import variograms as vrg

def main():

    x = np.array([9.7, 43.8]).T;
    y = np.array([47.6, 24.6]).T;
    v = [1.22, 2.822];

    xi = 18.8;
    yi = 67.9;
    
    print(vrg.ordinary_kriging(x,y,v,xi,yi,vario))
    
    
def vario(h):
    return 0.006*h + 0.1*(h>0)
if __name__ == '__main__':
    main()
