import keras.backend as K
from keras.engine import Model


class IWAE(object):
    def __init__(self, inpt, latents, reconstruction, likelihood, k_samples, prior_weight=1):  # noqa:E501
        self.__dict__.update(locals())

        # 'Model' is a trainable keras object
        self.model = Model(inpt, reconstruction)
        self.model.add_loss(self.elbo_loss())

    def set_samples(self, k):
        # TODO - use learning_phase() to set to 1 during testing?
        K.set_value(self.latent.k_samples, k)

    def elbo_loss(self):
        # Repeat inputs once per latent sample, then "fold" samples into batch dimension to match
        # shape of reconstruction.
        repeated_input = K.repeat(self.inpt, self.k_samples)
        shape = K.shape(repeated_input)
        new_shape = K.concatenate([[shape[0] * shape[1]], shape[2:]], axis=0)
        input_batch = K.reshape(repeated_input, new_shape)

        # NLL loss term is E_q(z|x) [ -log p(x|z) ] and has shape (batch, samples)
        nll_losses = self.likelihood.nll(input_batch, self.reconstruction)
        nll_losses = K.reshape(nll_losses, K.stack([-1, self.k_samples]))

        # Each KL loss term is E_q(z|x) [ log q(z|x) / p(z) ] and has shape (batch, samples)
        kl_losses = K.sum([latent.sample_kl() for latent in self.latents])

        # Total loss is simply sum of KL and NLL terms and has shape (batch, samples)
        total_loss = self.prior_weight * kl_losses + nll_losses

        # Final loss is weighted sum across k samples. More precisely, the total gradient is a
        # weighted sum of sample gradients. K.stop_gradient() is used to make the weights act on
        # gradients and not provide gradients themselves (weights are not 'learned' per se).
        # Weights have shape (batch, samples).
        weights = K.stop_gradient(self._get_weights())

        # Final loss per input is a weighted sum of sample losses
        return K.sum(total_loss * weights, axis=-1)

    def _get_weights(self):
        log_diff = K.sum([p.log_prob(q.samples) - q.log_prob(q.samples)
                          for (q, p) in zip(self.latents, self.priors)])
        weights_unnormalized = K.exp(log_diff)
        return weights_unnormalized / K.sum(weights_unnormalized, axis=-1, keepdims=True)


class VAE(IWAE):
    def _get_weights(self):
        weights_unnormalized = K.ones_like(self.latents[0].samples[:, :, 0])
        return weights_unnormalized / K.cast(self.k_samples, 'float32')
