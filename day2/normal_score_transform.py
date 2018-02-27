import numpy as np
from scipy.stats import norm

class transform():
    """
    Class for defining empirical normal score transform based on data.
    Attributes:
        z: ordered and processed data
        cdf: cumulative distribution function
        y: inverse cdf
    """
    def __init__(self, z):
        """
        Constructs empirical normal score transform for a given data.
        Arguments:
            z: array containing measurements
        """
        ordered_data = np.sort(z)
        extension = (ordered_data[1] - ordered_data[0]) /2
        ordered_data = np.insert(ordered_data, 0, ordered_data[0] -extension, )
        ordered_data = np.append(ordered_data, ordered_data[-1]+extension)
        n = ordered_data.shape[0]
        F = np.arange(n+1)/n

        ordered, i = np.unique(ordered_data, return_index=True)
        F = F[i]

        self.z = ordered
        self.cdf = F
        self.y = norm.ppf(F)

    def direct(self, z):
        """
        Returns normal score transformed data
        """
        u = np.interp(z, self.z, self.cdf)
        y = norm.ppf(u)
        return y

    def back(self, y):
        """
        Inverse normal score transform
        """
        u = norm.cdf(y)
        z = np.interp(u, self.cdf, self.z)
        return z
