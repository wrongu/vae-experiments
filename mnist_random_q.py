import os
import numpy as np
import matplotlib.pyplot as plt
from util import class_categorical
from models import gaussian_mnist
import keras.backend as K
from sys import argv

###################
# LOAD MNIST DATA #
###################

from my_mnist import x_train, x_test, y_test

img_rows, img_cols = 28, 28
img_pixels = img_rows * img_cols

###############################
# FITTING AND ENTROPY MEASURE #
###############################

model_type = argv[1]
latent_dim = 2
pixel_std = .05
k_samples = 16

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
prior_extent, prior_res = 3, 100
xs = np.linspace(-prior_extent, prior_extent, prior_res)
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
    vae = gaussian_mnist(model_type, latent_dim=latent_dim, pixel_std=pixel_std, k=k_samples)

    # Create model functions
    ll_fn = K.function([vae.inpt], [vae.nll])
    q = K.function([vae.inpt], [vae.latents[0].mean, vae.latents[0].log_var])

    # Load pre-trained weights if they exist, else train the model
    if os.path.exists(weights_file):
        vae.model.load_weights(weights_file)
    else:
        # Freeze Q(z|x) and train only P(x|z)
        last_latent = 0
        for i, l in enumerate(vae.model.layers):
            if l in vae.latents:
                last_latent = i
        for l in vae.model.layers[:last_latent + 1]:
            l.trainable = False

        # Train P model only
        vae.model.compile(loss=None, optimizer='rmsprop')
        vae.model.fit(x_train, shuffle=True, epochs=10, batch_size=100)

        # Save trained model to a file
        vae.model.save_weights(weights_file)

    # Reset to 1 sample for predictions
    vae.set_samples(1)

    # Compute class-entropy of q
    pred_mean, pred_log_var = q([x_test])
    pred_var = np.exp(pred_log_var)
    pred_std = np.sqrt(pred_var)
    categorical, extent = class_categorical(pred_mean, pred_std, y_test,
                                            res=prior_res, extent=prior_extent)
    entropy = -(categorical * np.log(categorical)).sum(axis=2)
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
