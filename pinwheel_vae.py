import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from vae import IWAE, IsoGaussianPrior, DiagonalGaussianLatent, GaussianLikelihood
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K

#############
# LOAD DATA #
#############

data_file = os.path.join("numpy_data", "pinwheel.npy")
data = np.load(data_file)
split = round(.9 * len(data))
x_train = data[:split, :]
x_test = data[split:, :]

#################
# CREATE MODELS #
#################

input_dim = 2
latent_dim = 1
likelihood_std = .01
k_samples = 10
n_models = 5
prior_weight = K.variable(0.0, name='prior_weight')


def new_vae():
    # Q MODEL
    q_model = Sequential()
    q_model.add(Dense(32, activation='relu', input_dim=input_dim))
    q_model.add(Dense(32, activation='relu', input_dim=input_dim))

    # LATENT
    latent = DiagonalGaussianLatent(dims=latent_dim)

    # PRIOR
    prior = IsoGaussianPrior(dims=latent_dim)

    # GENERATIVE MODEL
    p_model = Sequential()
    p_model.add(Dense(32, activation='relu', input_dim=latent_dim))
    p_model.add(Dense(32, activation='relu', input_dim=latent_dim))
    p_model.add(Dense(input_dim))

    # LIKELIHOOD
    likelihood = GaussianLikelihood(likelihood_std)

    # Combine the above parts into a single model
    vae = IWAE(q_model=q_model,
               latent=latent,
               prior=prior,
               p_model=p_model,
               likelihood=likelihood,
               k=k_samples,
               prior_weight=prior_weight)
    vae.model.compile(loss=None, optimizer='rmsprop')
    return vae

###########
# FITTING #
###########

vaes = [None] * n_models
for i in range(n_models):
    vaes[i] = new_vae()
    # Load pre-trained weights if they exist
    weights_file = os.path.join("models", "pinwheel_%d.h5" % i)
    if os.path.exists(weights_file):
        vaes[i].model.load_weights(weights_file)
    else:
        K.set_value(prior_weight, 0)
        vaes[i].model.fit(x_train, shuffle=True, epochs=15, batch_size=100, validation_data=(x_test, None))  # noqa:E501
        K.set_value(prior_weight, .4)
        vaes[i].model.fit(x_train, shuffle=True, epochs=15, batch_size=100, validation_data=(x_test, None))  # noqa:E501
        K.set_value(prior_weight, .8)
        vaes[i].model.fit(x_train, shuffle=True, epochs=15, batch_size=100, validation_data=(x_test, None))  # noqa:E501
        K.set_value(prior_weight, 1)
        vaes[i].model.fit(x_train, shuffle=True, epochs=30, batch_size=100, validation_data=(x_test, None))  # noqa:E501
        # Save trained model to a file
        vaes[i].model.save_weights(weights_file)

#################
# VISUALIZATION #
#################

for vae in vaes:
    vae.set_samples(1)
# q = K.function([vae.q_model.input], [vae.latent.mean])
# z_test = q([x_test])[0]
# plt.subplot(121)
# plt.scatter(x_test[:, 0], x_test[:, 1], marker='.')

# plt.subplot(122)
# plt.scatter(z_test[:, 0], z_test[:, 1], marker='.')

# plt.show()
zs = np.random.randn(1000, latent_dim)

zs_2d = np.zeros((1000, 2))
zs_2d[:, 0] = zs[:, 0]
zs_2d[:, 1] = zs[:, latent_dim - 1]

# Get a polar color scheme
hues = np.clip((np.arctan2(zs_2d[:, 0], zs_2d[:, 1]) + np.pi) / (2 * np.pi), 0, 1)
values = np.clip(np.sqrt(zs_2d[:, 0]**2 + zs_2d[:, 1]**2) / 2, 0, 1)
colors = hsv_to_rgb(np.array([hues, values, np.ones(hues.shape)]).T)

# Plot input space
ax = plt.subplot(1, 2, 1)
ax.set_aspect(1)
# plt.scatter(x_train[:, 0], x_train[:, 1], marker='.')
plt.hexbin(x_train[:, 0], x_train[:, 1], gridsize=30, extent=(-2, 2, -2, 2))

# Plot Z space
# plt.subplot(1, 2, 2)
# plt.scatter(zs_2d[:, 0], zs_2d[:, 1], c=colors, marker='.')

# Compute average density across models
density = np.zeros((1000, n_models, input_dim))
iters = 1
for iv, vae in enumerate(vaes):
    render = K.function([vae.latent_sample], [vae.reconstruction])
    # sample = K.function([vae.q_model.input], [vae.latent_sample])

    # Iterate on Q(z|x) ~ P(x|z)
    for i in range(iters):
        xs = render([zs])[0] + np.random.randn(*zs.shape) * likelihood_std
        # zs = sample([xs])[0]
    density[:, iv, :] = xs

density = density.reshape(-1, input_dim)
ax = plt.subplot(1, 2, 2)
ax.set_aspect(1)
# plt.scatter(xs[:, 0], xs[:, 1], c=colors, marker='.')
plt.hexbin(xs[:, 0], xs[:, 1], gridsize=30, extent=(-2, 2, -2, 2))

plt.show()
