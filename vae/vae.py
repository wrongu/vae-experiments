import keras.backend as K
from keras.engine.topology import Layer
from keras.engine import Model


class FoldSamplesIntoBatch(Layer):
    def call(self, x):
        input_shape = K.shape(x)
        new_shape = K.concatenate([[input_shape[0] * input_shape[1]], input_shape[2:]], axis=0)
        return K.reshape(x, new_shape)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            return (None,) + tuple(input_shape[2:])
        else:
            return (input_shape[0] * input_shape[1],) + tuple(input_shape[2:])


class IWAE(object):
    def __init__(self, q_model, latent, prior, p_model, likelihood, k=None, prior_weight=1):
        self.__dict__.update(locals())

        # latent.k_samples is a computation graph variable controlling the number (k) of samples
        # drawn from the latent per training example.
        if k is not None:
            K.set_value(latent.k_samples, k)

        # "Call" the latent layer on the output of the q_model
        self.expanded_latent_sample = latent(q_model.output)

        # Reshape the latent sample so that, as far as the p_model is concerned, the k samples are
        # simply a larger batch.
        self.latent_sample = FoldSamplesIntoBatch()(self.expanded_latent_sample)

        # Apply p_model to flattened samples
        self.reconstruction = p_model(self.latent_sample)

        # Reshape the reconstruction back into (batch, samples, ...)
        batch_sample_dims = K.shape(self.expanded_latent_sample)[:2]
        remaining_dims = K.prod(K.shape(self.reconstruction)[1:])
        new_shape = K.concatenate([batch_sample_dims, [remaining_dims]], axis=0)
        self.flat_reconstruction_samples = K.reshape(self.reconstruction, new_shape)

        # 'Model' is a trainable keras object
        self.model = Model(q_model.input, self.reconstruction)
        self.model.add_loss(self.elbo_loss())

    def set_samples(self, k):
        # TODO - use learning_phase() to set to 1 during testing?
        K.set_value(self.latent.k_samples, k)

    def elbo_loss(self):
        # KL loss term is E_q(z|x) [ log q(z|x) / p(z) ] and has shape (batch, samples) by means of
        # using the expanded_latent_sample rather than the latent_sample
        kl_losses = self.latent.log_prob(self.expanded_latent_sample) \
            - self.prior.log_prob(self.expanded_latent_sample)

        # NLL loss term is E_q(z|x) [ -log p(x|z) ] and has shape (batch, samples)
        input_batch = K.repeat(K.reshape(self.q_model.input, (-1, K.prod(K.shape(self.q_model.input)[1:]))), self.latent.k_samples)
        nll_losses = self.likelihood.nll(input_batch, self.flat_reconstruction_samples)

        # Total loss is simply sum of KL and NLL terms and has shape (batch, samples)
        total_loss = self.prior_weight * kl_losses + nll_losses

        # Total loss is weighted sum across k samples. More precisely, the total gradient is a
        # weighted sum of sample gradients. K.stop_gradient() is used to make the weights act on
        # gradients and not provide gradients themselves (weights are not 'learned' per se).
        # Weights have shape (batch, samples).
        weights = K.stop_gradient(self._get_weights())

        # Final loss per input is a weighted sum of sample losses
        return K.sum(total_loss * weights, axis=-1)

    def _get_weights(self):
        log_q = self.latent.log_prob(self.expanded_latent_sample)
        log_p = self.prior.log_prob(self.expanded_latent_sample)
        weights_unnormalized = K.exp(log_p - log_q)
        return weights_unnormalized / K.sum(weights_unnormalized, axis=-1, keepdims=True)


class VAE(IWAE):
    def _get_weights(self):
        weights_unnormalized = K.ones_like(self.expanded_latent_sample[:, :, 0])
        return weights_unnormalized / K.cast(self.latent.k_samples, 'float32')
