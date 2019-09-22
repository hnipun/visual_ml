import math
from numpy import ndarray


def gaussian(x, mean, sigma):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


class NaiveBayes:
    _X = ndarray
    _y = ndarray
    _n_samples = int
    _n_features = int

    def __init__(self, x: ndarray, y: ndarray):
        self._X = x
        self._y = y
        self._n_samples, self._n_features = x.shape

    def fit(self):
        

    def _naive_bayes(brightness, mean_varience):
        prob_0 = gaussian(brightness, brigtness_mean_0, math.sqrt(brigtness_vriance_0)) * gaussian(mean_varience,
                                                                                                   mean_varaince_mean_0,
                                                                                                   math.sqrt(
                                                                                                       mean_varaince_variance_0))
        prob_1 = gaussian(brightness, brigtness_mean_1, math.sqrt(brigtness_vriance_1)) * gaussian(mean_varience,
                                                                                                   mean_varaince_mean_1,
                                                                                                   math.sqrt(
                                                                                                       mean_varaince_variance_1))

        if prob_0 > prob_1:
            return 0
        return 1

    def _
