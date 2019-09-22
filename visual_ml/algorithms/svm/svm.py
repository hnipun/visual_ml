from numpy import ndarray, zeros, outer, dot, diag, ones, ravel, arange, sum
from matplotlib import pyplot as plt
from cvxopt import matrix, solvers
from typing import Dict, Optional, Type, List, Set


# move to your own QP wrapper later
class SVM:
    _X = ndarray
    _y = ndarray
    _n_samples = int
    _n_features = int

    def __init__(self, x: ndarray, y: ndarray):
        self._X = x
        self._y = y
        self._n_samples, self._n_features = x.shape

    def fit(self):
        # P = X^T X
        K = zeros((self._n_samples, self._n_samples))
        for i in range(self._n_samples):
            for j in range(self._n_samples):
                K[i, j] = dot(self._X[i], self._X[j])

        P = matrix(outer(self._y, self._y) * K)
        # q = -1 (1xN)
        q = matrix(ones(self._n_samples) * -1)
        # A = y^T
        A = matrix(y, (1, self._n_samples))
        # b = 0
        b = matrix(0.0)
        # -1 (NxN)
        G = matrix(diag(ones(self._n_samples) * -1))
        # 0 (1xN)
        h = matrix(zeros(self._n_samples))

        solution = solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        a = ravel(solution['x'])

        # Lagrange have non zero lagrange multipliers
        sv = a > 1e-5
        ind = arange(len(a))[sv]

        self.a = a[sv]
        self.sv = self._X[sv]
        self.sv_y = self._y[sv]

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weights
        self.w = zeros(self._n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]
