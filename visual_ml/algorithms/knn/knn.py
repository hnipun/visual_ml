import numpy as np
import heapq

from numpy import ndarray
from statistics import mode
from typing import Dict
from visual_ml.utilities import euclidean_distance


class KNN:
    _X = ndarray
    _y = ndarray
    _n_samples = int
    _n_features = int
    _k = int
    _items = list

    # TODO need a function to assign similarity dynamically according to the string
    def __init__(self, x: list, y: list, k: int):
        self._X = np.asarray(x)
        self._y = np.asarray(y)
        self._n_classes = len(set(y))
        self._k = k
        self._items = []

    def fit(self):
        for x, y in zip(self._X, self._y):
            self._items.append({'features': x, 'class': y})

    def predict(self, x: list):
        distances = []
        for item in self._items:
            distance = euclidean_distance(item['features'], x)
            distances = self._update_nearest_neighbors(distances, (-distance, item['class']))

        return mode([value[1] for value in distances])

    def _update_nearest_neighbors(self, heap, distance):
        if len(heap) < self._k:
            heapq.heappush(heap, distance)
        elif distance > heap[0]:
            heapq.heapreplace(heap, distance)

        return heap
