import numpy as np


def objective(X, y, theta):
    return np.sum(y ** theta.T.dot(X) - np.log(1 + np.exp(theta.T.dot(X))))


def gradient(X, y, theta):
    return np.sum(y.dot(X) - np.exp(theta.T.dot(X)).dot(X) / (1 + np.exp(theta.T.dot(X))))
