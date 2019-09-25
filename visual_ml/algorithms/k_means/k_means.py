import numpy as np
from numpy import ndarray
from visual_ml.utilities import array_almost_equal, euclidean_distance


class KMeans:
    _X = ndarray
    _K = int
    _iterations = int
    _centroids = ndarray
    _n_samples = int
    _n_features = int
    _cost = ndarray

    def __init__(self, x: list, k: int, centroids: list, iterations: int):
        self._X = np.asarray(x)
        self._K = k
        self._iterations = iterations
        self._centroids = np.asarray(centroids) if centroids else np.zeros([self._K, self._n_features])
        self._n_samples, self._n_features = self._X.shape
        self._clusters = np.zeros([self._n_features])
        self._cost = np.zeros([self._iterations])

    def _update_mean(self):
        """
        update the centroids
        :return:
        """
        centroids = np.zeros([self._K, self._n_features])
        counts = [0] * self._K
        for x, c in zip(self._X, self._clusters):
            centroids[c] = (centroids[c] * counts[c] + x) / (counts[c] + 1)
            counts[c] += 1

        return centroids

    def _find_clusters(self):
        """
        find the cluster of each data point given the centroids
        :return:
        """
        res = []
        for x in self._X:
            dist = []
            for c in self._centroids:
                dist.append(euclidean_distance(x, c))
            res.append(dist.index(min(dist)))

        return res

    def _sum_of_errors(self):
        """
        calculate the sum of errors. this is the objective of k means
        :return:
        """
        cost = 0
        for c in self._centroids:
            for x in self._X:
                cost += euclidean_distance(x, c)

        return cost

    def fit(self):
        """
        implementation of the k means algorithm using helper functions
        :return:
        """
        for i in range(self._iterations):
            self._cost[i] = self._sum_of_errors()
            self._clusters = self._find_clusters()
            new_centroids = self._update_mean()

            if all([array_almost_equal(0.0001, x, y) for x, y in zip(new_centroids, self._centroids)]):
                break
            else:
                self._centroids = new_centroids


if __name__ == '__main__':
    import random
    from scipy.io import loadmat

    data = loadmat('samples.mat')['AllSamples']
    centroids = []
    for c in range(7):
        centroids.append([random.randint(1, 10), random.randint(1, 10)])
    print(centroids)

    kmeans = KMeans(data, 7, centroids, 10)
    kmeans.fit()
