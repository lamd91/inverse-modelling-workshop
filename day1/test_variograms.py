import numpy as np
import variograms

def test_variogram_cloud():
    x=np.array([3.0])
    y=np.array([2.0])
    v=np.array([1.0])
    (hc, gc) = variograms.variogram_cloud(x,y,v)
    assert(hc.size == 0 and gc.size == 0)

def test_variogram_cloud_2():
    x=np.array([2.0, 3.0])
    y=np.array([2.0, 3.0])
    v=np.array([1.0, 2.0])
    (hc, gc) = variograms.variogram_cloud(x,y,v)
    assert(hc[0] == np.sqrt(2.0) and gc[0] == 0.5)

#maybe better check shape?
def test_variogram_cloud_3():
    n = 100;
    x=np.zeros(n)
    y=np.zeros(n)
    v=np.zeros(n)
    (hc, gc) = variograms.variogram_cloud(x,y,v)
    assert(hc.size == n*(n-1)/2 and gc.size == n*(n-1)/2)
