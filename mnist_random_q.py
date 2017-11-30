import os
import numpy as np
import matplotlib.pyplot as plt
from models import gaussian_mnist
import keras.backend as K
from sys import argv

###################
# LOAD MNIST DATA #
###################

from my_mnist import x_train, x_test, y_test

img_rows, img_cols = 28, 28
img_pixels = img_rows * img_cols

#################
# CLASS-ENTROPY #
#################


def get_class_multinomial(q, extent, resolution):
    # Apply recognition model to test set.
    pred_mean, pred_log_var = q([x_test])
    pred_var = np.exp(pred_log_var)

    # Create meshgrid of points covering the +/- maximum extent
    xs = np.linspace(-extent, extent, resolution)
    ys = np.linspace(-extent, extent, resolution)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.concatenate([np.expand_dims(xx, 2), np.expand_dims(yy, 2)], 2)

    # Create a multinomial distribution over classes at each point in the grid/image
    eps = 1e-10
    multinomial = np.full(xx.shape + (10,), eps)

    def multivariate_gaussian(pos, mu, sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        Source: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
        """

        n = mu.shape[0]
        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        d = np.sqrt((2 * np.pi)**n * sigma_det)
        # This einsum call calculates (x-mu)T.sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos - mu, sigma_inv, pos - mu)

        return np.exp(-fac / 2) / d

    # For each test point of class c, evaluate normpdf of the model and add to multinomial[:,:,c]
    for c, mu, var in zip(y_test, pred_mean, pred_var):
        multinomial[:, :, c] += multivariate_gaussian(pts, mu, np.diag(var))

    # Normalize across classes at each point
    return multinomial / multinomial.sum(axis=2)[:, :, np.newaxis]


###############################
# FITTING AND ENTROPY MEASURE #
###############################

model_type = argv[1]
n_models = 150
q_entropies = np.zeros((n_models,))
p_log_likelihoods = np.zeros((n_models,))

# Placeholder for results on trained model
q_trained_entropy = 0
p_trained_log_likelihood = 0

precomputed_file = os.path.join('numpy_data', 'mnist_random_q_results_%s.npz' % model_type)
if os.path.exists(precomputed_file):
    data = np.load(precomputed_file)
    q_entropies = data['q_entropies']
    p_log_likelihoods = data['p_log_likelihoods']

    # Expand if needed, truncate if not
    if len(q_entropies) != n_models:
        q_entropies.resize((n_models,))
        p_log_likelihoods.resize((n_models,))

# Get prior density
xs = np.linspace(-5, 5, 200)
xx, yy = np.meshgrid(xs, xs)
rr2 = xx**2 + yy**2
prior_p = np.exp(-rr2 / 2)
prior_p /= prior_p.sum()

for i in range(n_models + 1):
    # Model weights file for iteration i (or fully-trained model for final data point)
    weights_file = os.path.join("models", "mnist_random_q_%04d_%s.h5" % (i, model_type))
    if i == n_models:
        weights_file = os.path.join("models", "mnist_%s.h5" % model_type)

    # Skip if already computed
    if i < n_models and q_entropies[i] != 0 and os.path.exists(weights_file):
        continue

    # Compute q_entropies[i] and p_log_likelihoods[i] for a saved or new model
    print(i)
    vae = gaussian_mnist(latent_dim=2, pixel_std=.05, k=8)

    # Create model functions
    nll = vae.likelihood.nll(vae.q_model.input, vae.reconstruction)
    ll_fn = K.function([vae.q_model.input], [nll])
    q = K.function([vae.q_model.input], [vae.latent.mean, vae.latent.log_var])

    # Load pre-trained weights if they exist, else train the model
    if os.path.exists(weights_file):
        vae.model.load_weights(weights_file)
    else:
        # Freeze Q(z|x) and train only P(x|z)
        for layer in vae.q_model.layers:
            layer.trainable = False
        vae.latent.trainable = False

        # Train P model only
        vae.model.compile(loss=None, optimizer='rmsprop')
        vae.model.fit(x_train, shuffle=True, epochs=10, batch_size=100)

        # Save trained model to a file
        vae.model.save_weights(weights_file)

    # Reset to 1 sample for predictions
    vae.set_samples(1)

    # Compute class-entropy of q
    multinomial = get_class_multinomial(q, 5, 200)
    entropy = -(multinomial * np.log(multinomial)).sum(axis=2)
    if i < n_models:
        q_entropies[i] = (prior_p * entropy).sum()
    else:
        q_trained_entropy = (prior_p * entropy).sum()

    # Compute log-likelihood of p
    if i < n_models:
        p_log_likelihoods[i] = -ll_fn([x_test])[0].sum()
    else:
        p_trained_log_likelihood = -ll_fn([x_test])[0].sum()

    # Re-save results after every iteration
    np.savez(precomputed_file, q_entropies=q_entropies, p_log_likelihoods=p_log_likelihoods)

plt.figure()
plt.subplot(121)
plt.scatter(q_entropies, p_log_likelihoods)
plt.subplot(122)
plt.scatter(q_entropies, p_log_likelihoods)
plt.scatter(q_trained_entropy, p_trained_log_likelihood)
plt.show()
