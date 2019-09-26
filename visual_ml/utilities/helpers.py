import math


def euclidean_distance(x_, y_):
    """
    given two samples return the Euclidean distance between
    :param x_:
    :param y_:
    :return:
    """
    summation = 0
    for x, y in zip(x_, y_):
        summation += math.pow(x - y, 2)

    return math.sqrt(summation)


def array_almost_equal(epsilon, x_, y_):
    """
    return true if each element pairs of two arrays only epsilon apart
    :param epsilon: difference
    :param x_:
    :param y_:
    :return:
    """
    return all([abs(x - y) < epsilon for x, y in zip(x_, y_)])


def gaussian(x, mean, sigma):
    """
    return the probability for the given input x, for the distribution
    :param x:
    :param mean:
    :param sigma:
    :return:
    """
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

# need to add similarity measurements and objective functions
