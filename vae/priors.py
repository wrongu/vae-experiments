import numpy as np
import keras.backend as K


class Prior(object):
    def __init__(self, dim):
        self.d = dim


class IsoGaussianPrior(Prior):
    def log_prob(self, x):
        return -K.sum(x * x, axis=-1) / 2

    def sample(self, n):
        return K.random_normal(shape=(n, self.d))


class DiscreteUniformPrior(Prior):
    def log_prob(self, x):
        return K.constant(-np.log(self.d))

    def sample(self, n):
        random_indices = K.cast(K.random_uniform(shape=(n,), maxval=self.d), 'int32')
        return K.one_hot(random_indices, self.d)
