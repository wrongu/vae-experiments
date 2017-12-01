from keras.engine.topology import Layer
from tensorflow.contrib.distributions import OneHotCategorical
from .priors import IsoGaussianPrior, DiscreteUniformPrior
import keras.backend as K


class FoldSamplesIntoBatch(Layer):
    def call(self, x):
        input_shape = K.shape(x)
        new_shape = K.concatenate([[input_shape[0] * input_shape[1]], input_shape[2:]], axis=0)
        return K.reshape(x, new_shape)

    def compute_output_shape(self, input_shape):
        print("FOLD SHAPE", input_shape)
        if input_shape[0] is None:
            return (None,) + tuple(input_shape[2:])
        else:
            return (input_shape[0] * input_shape[1],) + tuple(input_shape[2:])


class Latent(Layer):
    def __init__(self, dim, prior, k_samples=1, **kwargs):
        super(Latent, self).__init__(**kwargs)

        # Convert k_samples to keras variable (unless it already is one)
        # try:
        #     self.k_samples = K.variable(int(k_samples), name="k_samples", dtype=np.int32)
        # except TypeError:
        self.k_samples = k_samples

        # Record instance variables
        self.d = dim
        self.prior = prior(dim)

    def sample_kl(self):
        try:
            return K.tile(K.expand_dims(self.analytic_kl()), (1, self.k_samples))
        except ValueError:
            # Use monte carlo kl estimate
            return self.log_prob(self.samples) - self.prior.log_prob(self.samples)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.k_samples, self.d)

    def call(self, x):
        # Must be implemented by subclass
        raise NotImplementedError()

    def log_prob(self, x):
        # Must be implemented by subclass
        raise NotImplementedError()

    def analytic_kl(self):
        raise ValueError()


class DiagonalGaussianLatent(Latent):
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
        eps = K.random_normal(shape=sample_shape, mean=0., stddev=1.0)

        # Shape of self.samples is (batch, samples, dim)
        self.samples = rep_mean + eps * rep_std
        # Shape of self.flat_samples is (batch * samples, dim)
        self.flat_samples = FoldSamplesIntoBatch()(self.samples)

        return self.flat_samples

    def log_prob(self, x):
        x_samples = K.shape(x)[1]
        variance = K.repeat(K.exp(self.log_var), x_samples)
        log_det = K.sum(K.repeat(self.log_var, x_samples), axis=-1)
        x_diff = x - K.repeat(self.mean, x_samples)
        return -(K.sum((x_diff / variance) * x_diff, axis=-1) + log_det) / 2

    def analytic_kl(self):
        if isinstance(self.prior, IsoGaussianPrior):
            # In general for two multi-variate normals
            #   kl(p1||p2)=[log(det(C2)/det(C1)) - dim + Tr(C2^-1*C1) + (m2-m1).T*C2^-1*(m2-m1)]/2
            # where C1 and C2 are covariances, and m1 and m2 are means. Since 'IsoGaussianPrior' is
            # mean 0 and identity covariance, this is simplified significantly:
            #   kl(p1||iso)=[-log(det(C1)) - dim + Tr(C1) + m1.T*m1]/2
            log_det_p1 = K.sum(self.log_var, axis=-1)
            trace_c1 = K.sum(K.exp(self.log_var), axis=-1)
            mean_sq_norm = K.sum(self.mean**2, axis=-1)
            return (-log_det_p1 - self.d + trace_c1 + mean_sq_norm) / 2
        else:
            raise ValueError("Prior must be IsoGaussianPrior")


class CategoricalLatent(Latent):
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.dense_logits = self.add_weight(shape=(input_dim, self.d),
                                            name='latent_mean_kernel',
                                            initializer='glorot_uniform')
        self.built = True

    def call(self, x):
        self.logits = K.dot(x, self.dense_logits)

        # Create sample from the latent distribution. Note that the TF OneHotCategorical
        # distribution returns shape (samples, batch, dim) with integer types but we want (batch,
        # samples, dim) with float type.
        ohc_sample = OneHotCategorical(logits=self.logits).sample(self.k_samples)
        self.samples = K.cast(K.permute_dimensions(ohc_sample, (1, 0, 2)), 'float32')
        self.flat_samples = FoldSamplesIntoBatch()(self.samples)

        # Samples of shape (batch, samples, ...) are saved in self.samples. For input into the
        # generative model, 'fold' the samples into the batch dimension
        return self.flat_samples

    def log_prob(self, x):
        x_samples = K.shape(x)[1]
        return OneHotCategorical(logits=K.repeat(self.logits, x_samples)).log_prob(x)

    def sample_kl(self):
        if isinstance(self.prior, DiscreteUniformPrior):
            # p is unnormalized multinomial probability
            p = K.exp(self.logits)
            # partition is normalization constant (partition function)
            partition = K.sum(p, axis=-1, keepdims=True)
            return K.sum(p * self.logits, axis=-1) / partition - K.log(partition) + K.log(self.d)
        else:
            raise ValueError("Prior must be DiscreteUniformPrior")
