from keras.engine.topology import Layer
import keras.backend as K


class GaussianLikelihood(Layer):
    def __init__(self, sigma):
        self.sigma = sigma

    def nll(self, x_true, x_pred):
        return K.sum(K.square(x_true - x_pred) / (2 * self.sigma**2), axis=-1)
