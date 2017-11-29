import numpy as np
import keras.backend as K


class GaussianPrior(object):
    def __init__(self, dims=2):
        self.d = dims

    def log_prob(self, x):
        return -K.sum(x * x, axis=-1) / 2


class DiscreteUniformPrior(object):
    def __init__(self, dims):
        self.d = dims

    def log_prob(self, x):
        return K.constant(-np.log(self.d))
