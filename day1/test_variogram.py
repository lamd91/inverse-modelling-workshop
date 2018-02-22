"""
Test module of variogram to be executed by pytest
"""

import numpy as np

import variogram


def test_variogram_cloud():
    x = np.array([3.0])
    y = np.array([2.0])
    v = np.array([1.0])
    (hc, gc) = variogram.cloud(x, y, v)
    assert(hc.size == 0 and gc.size == 0)


def test_variogram_cloud_2():
    x = np.array([2.0, 3.0])
    y = np.array([2.0, 3.0])
    v = np.array([1.0, 2.0])
    (hc, gc) = variogram.cloud(x, y, v)
    assert(hc[0] == np.sqrt(2.0) and gc[0] == 0.5)


def test_variogram_cloud_3():
    n = 100
    x = np.zeros(n)
    y = np.zeros(n)
    v = np.zeros(n)
    (hc, gc) = variogram.cloud(x, y, v)
    assert(hc.shape[0] == n * (n - 1) / 2 and gc.shape[0] == n * (n - 1) / 2)
