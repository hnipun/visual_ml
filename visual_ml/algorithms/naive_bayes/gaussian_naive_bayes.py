import math
import numpy as np

from numpy import ndarray
from visual_ml.utilities import gaussian
from visual_ml.mle import estimate_1_d


# https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian

class NaiveBayes:
    _X = ndarray
    _y = ndarray
    _n_samples = int
    _n_features = int
    _priors: ndarray

    def __init__(self, x: list, y: list, priors: list):
        self._X = np.asarray(x)
        self._y = np.asarray(y)
        self._n_samples, self._n_features = self._X
        self._n_classes = len(set(y))
        self._priors = np.asarray(priors) if priors else self._estimate_priors()
        self.cond_feature_prob = {}

    def fit(self):
        for y, X in self._get_class_data().items():
            self.cond_feature_prob[y] = (estimate_1_d(X))

    def _get_class_data(self):
        class_data = {}

        for x, y in zip(self._X, self._y):
            class_data[y].add(x)

        return class_data

    def _estimate_priors(self):
        priors = counts = np.zeros([self._n_classes])

        for i, y in enumerate(self._y):
            priors[i] = (priors[i] * priors[i] + y) / (priors[i] + 1)
            counts[i] += 1

        return priors

    def predict(self, x):
        probabilities = []
        for y, dist in self._get_class_data().items():
            probabilities.append()