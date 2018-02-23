"""
Tests for variogram models. Can be run in pytest framework.
"""

import numpy as np

import model


def test_gaussian():
    h = 5
    sill = 1
    range = 5
    assert model.gaussian(h, sill, range) == 1 - np.exp(-3)


def test_exponential():
    h = 1
    sill = 1
    range = -3
    assert model.exponential(h, sill, range) == 1 - np.exp(1)


def test_sinus_cardinal():
    h = 1
    sill = 1
    range = 2
    assert model.sinus_cardinal(h, sill, range) == 1 - 2 * np.sin(0.5)


def test_hyperbolic():
    h = 1
    sill = 1
    range = -2
    assert model.hyperbolic(h, sill, range) == 2


def test_nugget():
    h = 1
    nugget = 2
    assert model.nugget(h, nugget) == 2


def test_spherical():
    h = 2
    sill = 1
    range = 3
    assert model.spherical(h, sill, range) == 1 - 0.5 * (2 / 3)**3


def test_linear():
    h = 2
    sill = 1
    range = 3
    assert model.linear(h, sill, range) == 2 / 3


def test_stable():
    h = 1
    sill = 1
    range = 4
    assert model.stable(h, sill, range) == 1 - np.exp(-3 / 2)
