from keras.engine.topology import Layer
from tensorflow.contrib.distributions import OneHotCategorical
import numpy as np
import keras.backend as K


class GaussianLatent(Layer):
    def __init__(self, dims=2, k=1, **kwargs):
        self.d = dims
        self.k_samples = K.variable(k, name="k_samples", dtype=np.int32)
        super(GaussianLatent, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.dense_mean = self.add_weight(shape=(input_dim, self.d),
                                          name='latent_mean_kernel',
                                          initializer='glorot_uniform')
        self.dense_log_var = self.add_weight(shape=(input_dim, self.d),
                                             name='latent_log_var_kernel',
                                             initializer='glorot_uniform')
        self.built = True

    def call(self, x):
        # Create dense connections to mean and log variance parameters
        self.mean = K.dot(x, self.dense_mean)
        self.log_var = K.dot(x, self.dense_log_var)

        # Repeat mean and variance self.k_samples times, once for each sample.
        rep_mean = K.repeat(self.mean, self.k_samples)
        rep_std = K.repeat(K.exp(self.log_var / 2), self.k_samples)

        # Create (reparameterized) sample from the latent distribution
        sample_shape = (K.shape(self.mean)[0], self.k_samples, self.d)
        self.eps = K.random_normal(shape=sample_shape, mean=0., stddev=1.0)
        return rep_mean + self.eps * rep_std

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.k_samples, self.d)

    def log_prob(self, x):
        variance = K.repeat(K.exp(self.log_var), self.k_samples)
        log_det = K.sum(K.repeat(self.log_var, self.k_samples), axis=-1)
        x_diff = x - K.repeat(self.mean, self.k_samples)
        return -(K.sum((x_diff / variance) * x_diff, axis=-1) + log_det) / 2

    def kl_loss(self):
        # In general for two multi-variate normals, we'd have
        #   kl(p1||p2) = [log(det(C2)/det(C1)) - dim + Tr(C2^-1*C1) + (m2-m1).T*C2^-1*(m2-m1)] / 2
        # where C1 and C2 are covariances, and m1 and m2 are means.
        # Note: p being normal(0,eye) and C1 being diagonal simplifies the following considerably
        log_det_p2_p1 = -K.sum(self.log_var, axis=-1)
        trace_c2_inv_c1 = K.sum(K.exp(self.log_var), axis=-1)
        mean_norm = K.sqrt(K.sum(self.mean**2, axis=-1))
        return (log_det_p2_p1 - self.d + trace_c2_inv_c1 + mean_norm) / 2


class CategoricalLatent(Layer):
    def __init__(self, dims=2, k=1, **kwargs):
        self.d = dims
        self.k_samples = K.variable(k, name="k_samples", dtype=np.int32)
        super(CategoricalLatent, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.dense_logits = self.add_weight(shape=(input_dim, self.d),
                                            name='latent_mean_kernel',
                                            initializer='glorot_uniform')
        self.built = True

    def call(self, x):
        self.logits = K.dot(x, self.dense_logits)

        # Create sample from the latent distribution. Note that TF distributions return shape
        # (samples, batch, dim) but we want (batch, samples, dim).
        samples = OneHotCategorical(logits=self.logits).sample(self.k_samples)
        return K.cast(K.permute_dimensions(samples, (1, 0, 2)), 'float32')

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.k_samples, self.d)

    def log_prob(self, x):
        return OneHotCategorical(logits=K.repeat(self.logits, self.k_samples)).log_prob(x)

    def kl_loss(self):
        # p is unnormalized multinomial probability
        p = K.exp(self.logits)
        # Z is normalization constant
        Z = K.sum(p, axis=-1)
        return K.sum(p * self.logits, axis=-1) / Z - K.log(Z) + np.log(self.d)
