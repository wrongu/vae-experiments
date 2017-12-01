import keras.backend as K
from keras.engine import Model


class IWAE(object):
    def __init__(self, inpt, latents, reconstruction, likelihood, k_samples, kl_weight=1):  # noqa:E501
        self.__dict__.update(locals())

        # 'Model' is a trainable keras object
        self.model = Model(inpt, reconstruction)
        self.model.add_loss(self.elbo_loss())

    def set_samples(self, k):
        # TODO - use learning_phase() to set to 1 during testing?
        K.set_value(self.k_samples, k)

    def elbo_loss(self):
        # Repeat inputs once per latent sample to size (batch, samples, -1), where -1 stands for
        # 'all subsequent dimensions flattened'.
        input_shape = K.shape(self.inpt)
        repeated_input = K.repeat(K.batch_flatten(self.inpt), self.k_samples)

        # NLL loss term is E_q(z|x) [ -log p(x|z) ] and has shape (batch, samples)
        batch_sample_shape = K.stack([input_shape[0], self.k_samples, -1])
        batch_sample_reconstruction = K.reshape(self.reconstruction, batch_sample_shape)
        self.nll = self.likelihood.nll(repeated_input, batch_sample_reconstruction)

        # Each KL loss term is E_q(z|x) [ log q(z|x) / p(z) ] and has shape (batch, samples)
        kl_losses = K.sum([latent.sample_kl() for latent in self.latents], axis=0)

        # Total loss is simply sum of KL and NLL terms and has shape (batch, samples)
        total_loss = self.kl_weight * kl_losses + self.nll

        # Final loss is weighted sum across k samples. More precisely, the total gradient is a
        # weighted sum of sample gradients. K.stop_gradient() is used to make the weights act on
        # gradients and not provide gradients themselves (weights are not 'learned' per se).
        # Weights have shape (batch, samples).
        weights = K.stop_gradient(self._get_weights())

        # Final loss per input is a weighted sum of sample losses
        return K.sum(total_loss * weights, axis=-1)

    def _get_weights(self):
        log_likelihood = -self.nll
        log_p = [p.log_prob(q.samples) for (p, q) in zip(self.priors, self.latents)]
        log_q = [q.log_prob(q.samples) for q in self.latents]
        weights_unnormalized = K.exp(log_likelihood + K.sum(log_p) - K.sum(log_q))
        return weights_unnormalized / K.sum(weights_unnormalized, axis=-1, keepdims=True)


class VAE(IWAE):
    def _get_weights(self):
        weights_unnormalized = K.ones_like(self.latents[0].samples[:, :, 0])
        return weights_unnormalized / K.cast(self.k_samples, 'float32')
