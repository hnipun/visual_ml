import math
import numpy as np
from numpy import ndarray


class KMeans:
    _X = ndarray
    _K = int
    _iterations = int
    _n_samples = int
    _n_features = int

    def __init__(self, x: ndarray, k: int, iterations: int):
        self._X = x
        self._K = k
        self._iterations = iterations
        self._n_samples, self._n_features = x.shape
        self._centroids = np.zeros([self._K, self._n_features])
        self._clusters = np.zeros([self._n_features])

    def _euclidean_distance(self, x_, y_):
        """
        given two samples return the Euclidean distance between
        :param x_:
        :param y_:
        :return:
        """
        summation = 0
        for x, y in zip(x_, y_):
            summation = math.pow(x - y, 2)

        return math.sqrt(summation)

    def _initialize_centroids(self, centroids):
        self._centroids = centroids

    def _update_mean(self):
        centroids = np.zeros([self._K, self._n_features])
        counts = [0] * self._K
        for (x, y), c in zip(D, self._centroids):
            _x, _y = centroids[c]
            centroids[c] = ((_x * counts[c] + x) / (counts[c] + 1), (_y * counts[c] + y) / (counts[c] + 1))
            counts[c] += 1

        return centroids

    def _find_clusters(self, centroids):
        res = []
        for x in self._X:
            dist = []
            for c in centroids:
                dist.append(self._euclidean_distance(x, c))
            res.append(dist.index(min(dist)))

        return res

    def _k_means(self):
        pass
