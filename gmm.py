import os
import numpy as np
from util import get_class_colors
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
from keras.layers import Input, Dense
from keras.models import Model
from keras.engine.topology import Layer
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
import keras.backend as K
import matplotlib.pyplot as plt


f = np.load(os.path.join("numpy_data", "gmm.npz"))
latent_dim = len(f['pis'])

data = {}
for key in f.keys():
    data[key] = f[key].astype(np.float32)

modes = [None] * latent_dim
for i in range(latent_dim):
    modes[i] = MultivariateNormalFullCovariance(loc=data['means'][i, ...],
                                                covariance_matrix=data['covariances'][i, ...])

data_input = Input(shape=(2,))
q_model = Dense(16, activation='relu')(data_input)
q_model = Dense(16, activation='relu')(q_model)
q_model = Dense(16, activation='relu')(q_model)
q_model = Dense(latent_dim)(q_model)


class ElboLayer(Layer):
    def __init__(self, input_tensor, **kwargs):
        super(ElboLayer, self).__init__(**kwargs)
        self._input = input_tensor

    def call(self, logits):
        norm_logits = logits - K.tile(K.logsumexp(logits, axis=-1, keepdims=True),
                                      (1, K.shape(logits)[1]))
        categorical = K.softmax(logits)
        kl = -K.sum(categorical * (norm_logits - K.log(data['pis'])), axis=-1)
        ll = K.transpose(K.stack([mode.log_prob(self._input) for mode in modes]))
        ll = K.sum(categorical * ll, axis=-1)
        elbo = ll - kl
        self.add_loss(-elbo, inputs=logits)
        return logits

elbo = ElboLayer(data_input)(q_model)
model = Model(data_input, elbo)
model.compile(optimizer='rmsprop', loss=None)

model.fit(data['data'],
          shuffle=True,
          epochs=50,
          batch_size=100)


###################
#  VISUALIZATION  #
###################

ext = 10
colors = get_class_colors(latent_dim)

ax = plt.subplot(141)
for i in range(latent_dim):
    pts = data['labels'] == i
    plt.scatter(data['data'][pts, 0], -data['data'][pts, 1], marker='.', c=colors[i])
plt.xlim([-ext, ext])
plt.ylim([-ext, ext])
plt.xticks([-ext, 0, ext])
plt.yticks([-ext, 0, ext])
ax.set_aspect(1)

plt.subplot(142)
xs = np.linspace(-ext, ext, 501)
xx, yy = np.meshgrid(xs, xs)
predictors = np.zeros((xx.size, 2))
predictors[:, 0] = xx.ravel()
predictors[:, 1] = yy.ravel()

Q = K.function([data_input], [model.output])
logits = Q([predictors])[0].reshape(xx.shape + (latent_dim,))
max_class = np.argmax(logits, axis=2)
class_image = colors[max_class]
plt.imshow(class_image, extent=(-ext, ext, -ext, ext))
plt.xticks([-ext, 0, ext])
plt.yticks([-ext, 0, ext])

plt.subplot(143)
logits = logits - logsumexp(logits, axis=2)[:, :, np.newaxis]
dist = np.exp(logits) + 1e-10
dist = dist / dist.sum(axis=2)[:, :, np.newaxis]
entropy = -(dist * logits).sum(axis=2)
plt.imshow(entropy, extent=(-ext, ext, -ext, ext))
plt.xticks([-ext, 0, ext])
plt.yticks([-ext, 0, ext])

plt.subplot(144)
true_log_posterior = np.zeros(xx.shape + (latent_dim,))
for i in range(latent_dim):
    mvn = multivariate_normal(mean=data['means'][i, :],
                              cov=data['covariances'][i, ...].reshape((2, 2)))
    true_log_posterior[:, :, i] = np.log(data['pis'][i]) + \
        np.array([mvn.logpdf(p) for p in predictors]).reshape(xx.shape)
true_log_posterior = true_log_posterior - \
    logsumexp(true_log_posterior, axis=2)[:, :, np.newaxis]
true_cat_pdf = np.exp(true_log_posterior)
true_cat_pdf = true_cat_pdf / true_cat_pdf.sum(axis=2)[:, :, np.newaxis]
kl = (true_cat_pdf * (true_log_posterior - np.log(dist))).sum(axis=2)
plt.imshow(kl, extent=(-ext, ext, -ext, ext), cmap='gray', vmin=0)
plt.xticks([-ext, 0, ext])
plt.yticks([-ext, 0, ext])
plt.colorbar()

plt.show()
