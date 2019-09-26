import numpy as np
from numpy import ndarray
from typing import Callable


class GradientDecent:
    _X = ndarray
    _y = ndarray
    _theta = ndarray
    _iterations = int
    _learning_rate = float

    def __init__(self, x: list, y: list, iterations: int, learning_rate: float, objective_function: Callable,
                 gradient_function: Callable):
        self._X = np.asarray(x)
        self._y = np.asarray(y)
        self._n_samples, self._n_features = self._X
        self._theta = np.zeros([self._n_features])
        self._iterations = iterations
        self._learning_rate = learning_rate
        self._objective_function = objective_function
        self._gradient_function = gradient_function

    def gradient_decent(self):
        cost_history = np.zeros(self._iterations)
        theta_history = np.zeros((self._iterations, self._n_features))

        for i in range(self._iterations):
            self._theta = self._theta - self._learning_rate * self._gradient_function(self._X, self._y, self._theta)
            cost_history[i] = self._objective_function(self._X, self._y, self._theta)
            theta_history[i, :] = self._theta.T

        return cost_history, theta_history
