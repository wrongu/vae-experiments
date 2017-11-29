import os
import numpy as np
import matplotlib.pyplot as plt
from util import get_class_colors
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
# CREATE MODELS #
#################

model_type = argv[1]
latent_dim = 2
pixel_std = .05
k_samples = 32

vae = gaussian_mnist(model_type, latent_dim=latent_dim, pixel_std=pixel_std, k=k_samples)
vae.model.compile(loss=None, optimizer='rmsprop')

###########
# FITTING #
###########

# Load pre-trained weights if they exist
weights_file = os.path.join("models", "mnist_entropy_%s.h5" % model_type)
if os.path.exists(weights_file):
    vae.model.load_weights(weights_file)
else:
    vae.model.fit(x_train,
                  shuffle=True,
                  epochs=15,
                  batch_size=100,
                  validation_data=(x_test, None))
    # Save trained model to a file
    vae.model.save_weights(weights_file)

###############################
# CLASS-ENTROPY VISUALIZATION #
###############################


# Get color for each numeric class
class_colors = get_class_colors(10)

# Get keras function of recognition model.
q = K.function([vae.q_model.input], [vae.latent.mean, vae.latent.log_var])

# Apply recognition model to test set.
pred_mean, pred_log_var = q([x_test])
pred_var = np.exp(pred_log_var)
pred_std = np.sqrt(pred_var)

# Compute maximum extent of predictions (3 std dev) in x and y direction
max_extent_x, max_extent_y = np.max(np.abs(pred_mean) + 3 * pred_std, axis=0)
max_extent_x, max_extent_y = [max(max_extent_x, max_extent_y)] * 2

# Create meshgrid of points covering the +/- maximum extent
xs = np.linspace(-max_extent_x, max_extent_x, 100)
ys = np.linspace(-max_extent_y, max_extent_y, 100)
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
multinomial = multinomial / multinomial.sum(axis=2)[:, :, np.newaxis]

# Find what class is most (plurality) represented at each point
max_class = multinomial.argmax(axis=2)

# Translate from max_class into a color at each point
class_image = class_colors[max_class]

# Get entropy across each multinomial (scaled to be in [0, 1])
relative_entropy = -np.sum(multinomial * np.log(multinomial), axis=2) / np.log(10)
gray_image = np.full(class_image.shape, .5)
entropy_image = gray_image * relative_entropy[:, :, np.newaxis] + \
    class_image * (1 - relative_entropy[:, :, np.newaxis])

plt.figure()
plt.subplot(131)
plt.imshow(class_image, extent=(-max_extent_x, max_extent_x, -max_extent_y, max_extent_y))
plt.title('Maximum class in latent space')
plt.subplot(132)
plt.imshow(relative_entropy, extent=(-max_extent_x, max_extent_x, -max_extent_y, max_extent_y))
plt.title('Entropy over classes')
plt.subplot(133)
plt.imshow(entropy_image, extent=(-max_extent_x, max_extent_x, -max_extent_y, max_extent_y))
plt.title('Adjusted class in latent space')
plt.show()

#################
# VISUALIZATION #
#################

render_range = 5
grid = 15
vae.set_samples(1)
render = K.function([vae.latent_sample], [vae.reconstruction])
full_image = np.zeros((img_rows * grid, img_cols * grid))
xs = np.linspace(-render_range, render_range, grid)
xx, yy = np.meshgrid(xs, xs)
inputs = np.vstack([yy.ravel(), xx.ravel()]).T
outputs = render([inputs])[0]
for ii in range(grid**2):
    i, j = divmod(ii, grid)
    full_image[j * img_cols:(j + 1) * img_cols, i * img_rows:(i + 1) * img_rows] = \
        np.reshape(outputs[ii], (img_rows, img_cols))
plt.imshow(full_image, cmap='Greys_r',
           extent=(-render_range, render_range, -render_range, render_range))
plt.show()
