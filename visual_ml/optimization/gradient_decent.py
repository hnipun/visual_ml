import numpy as np
from numpy import ndarray


class GradientDecent:
    _X = ndarray
    _y = ndarray
    _theta = ndarray
    _iterations = int

    def __init__(self, x: list, y: list, iterations: list, objective ):
        self._X = np.asarray(x)
        self._y = np.asarray(y)
        self._n_samples, self._n_features = self._X
        self._theta = np.zeros([self._n_features])
        self._iterations = iterations

    def _calculate_cost(self):
        predictions = self._X.dot(self._theta)
