import numpy as np
import cross_validation
import model

def model_function(h):
    return model.exponential(h, 14.5, 1.7) + model.nugget(h, 0.5)

def model_function2(h):
    return model.exponential(h, 12.0, 1.4) + model.nugget(h, 1.3)

def test():
    jura_data = np.genfromtxt('data.txt', names=True)
    x, y, v = jura_data['X'], jura_data['Y'], jura_data['Co']
    success, Q1, Q2, MSE, cR = cross_validation.orthonormal_residuals(x, y, v, model_function)
    assert(success == False)
    #result = (-0.05, 1.25, 5.96, 4.14)
    #np.testing.assert_allclose((Q1, Q2, MSE, cR), result, rtol=1e0)

def test2():
    jura_data = np.genfromtxt('data.txt', names=True)
    x, y, v = jura_data['X'], jura_data['Y'], jura_data['Co']
    success, Q1, Q2, MSE, cR = cross_validation.orthonormal_residuals(x, y, v, model_function2)
    assert(success == True)
    #result = (0.01, 0.96, 6.06, 5.38)
    #np.testing.assert_allclose((Q1, Q2, MSE, cR), result, rtol=1e0)
