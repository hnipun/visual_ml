import numpy as np


def objective_least_squares(X, y, theta):
    return (1 / 2 * len(y)) * np.sum(np.square(X.dot(theta) - y))


def gradient_least_squares(X, y, theta):
    return (1 / len(y)) * X.T.dot((theta.T.dot(X) - y))
