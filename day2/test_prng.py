"""
Tests for prng module
"""

import numpy as np

import prng

def test_congruential():
    n = 100000
    seed = 1231
    m = 2**31-1
    a = 48271
    c = 0
    u_pm = prng.congruential(n, seed, m, a, c)
    # check if mean and variance are ok
    np.testing.assert_allclose( (np.mean(u_pm), np.var(u_pm)),
                                    (0.5, 1/12),
                                    atol=1e-2 )

def test_gaussian():
    n = 100000
    seed = 1231
    u_g = prng.gaussian(n, seed)
    # check if mean and variance are ok
    np.testing.assert_allclose( (np.mean(u_g), np.var(u_g)),
                                    (0.0, 1.0),
                                    atol=1e-2 )
