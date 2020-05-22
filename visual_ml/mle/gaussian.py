import numpy as np

from numpy import ndarray


def estimate_1_d(array: ndarray):
    return np.mean(array), np.var(array)
