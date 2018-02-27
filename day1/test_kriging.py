"""
Pytest tests for kriging module
"""

import numpy as np

import kriging


def test_G_matrix():
    x = np.array([9.7, 43.8])
    y = np.array([47.6, 24.6])
    result = np.array([[0, -0.3468, 1], [-0.3468, 0, 1], [1, 1, 0]])
    print(kriging._G_matrix(x, y, model))
    assert(np.allclose(kriging._G_matrix(x, y, model), result, rtol=1e-4))


def test_g_vector():
    x = np.array([9.7, 43.8])
    y = np.array([47.6, 24.6])
    xi = 18.8
    yi = 67.9
    result = np.array([-0.2335, -0.4, 1])
    print(kriging._g_vector(x, y, xi, yi, model))
    assert(np.allclose(kriging._g_vector(x, y, xi, yi, model),
                       result, rtol=1e-4))


def test_ordinary():
    x = np.array([9.7, 43.8])
    y = np.array([47.6, 24.6])
    v = np.array([1.22, 2.822])
    xi = 18.8
    yi = 67.9
    result_v_est = 1.636390
    result_v_var = 0.420099
    print(kriging.ordinary(x, y, v, xi, yi, model))
    assert(np.allclose(kriging.ordinary(x, y, v, xi, yi, model),
                       (result_v_est, result_v_var), rtol=1e-4))


def test_ordinary_mesh():
    x = np.array([9.7, 43.8])
    y = np.array([47.6, 24.6])
    v = np.array([1.22, 2.822])
    xi = 18.8
    yi = 67.9
    result_v_est = 1.636390
    result_v_var = 0.420099
    print(kriging.ordinary_mesh(x, y, v, np.array([xi]),
                                np.array([yi]), model))
    assert(np.allclose(kriging.ordinary(x, y, v, xi, yi, model),
                       (result_v_est, result_v_var), rtol=1e-4))

def test_simple():
    x = np.array([9.7, 43.8])
    y = np.array([47.6, 24.6])
    v = np.array([1.22, 2.822])
    xi = 18.8
    yi = 67.9
    result_v_est = 3.3072001153402542
    result_v_var = -0.53863898500576712
    print(kriging.simple(x, y, v, xi, yi, model, 0))
    assert(np.allclose(kriging.simple(x, y, v, xi, yi, model, 0),
                       (result_v_est, result_v_var), rtol=1e-4))


def model(h):
    """
    Model test function
    """
    return 0.006 * h + 0.1 * (h > 0)
