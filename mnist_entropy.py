import os
import numpy as np
import matplotlib.pyplot as plt
from util import get_class_colors, class_categorical
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
weights_file = os.path.join("models", "mnist_%s.h5" % model_type)
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

# Evaluate distribution over a set of points in the latent space -- get a categorical distribution
# over classes at each point
categorical, latent_extent = class_categorical(pred_mean, pred_std, y_test, 10, res=100)

# Find what class is most (plurality) represented at each point
max_class = categorical.argmax(axis=2)

# Translate from max_class into a color at each point
class_image = class_colors[max_class]

# Get entropy across each categorical (scaled to be in [0, 1])
relative_entropy = -np.sum(categorical * np.log(categorical), axis=2) / np.log(10)
gray_image = np.full(class_image.shape, .5)
entropy_image = gray_image * relative_entropy[:, :, np.newaxis] + \
    class_image * (1 - relative_entropy[:, :, np.newaxis])

plt.figure()
plt.subplot(131)
plt.imshow(class_image, extent=latent_extent)
plt.title('Maximum class in latent space')
plt.subplot(132)
plt.imshow(relative_entropy, extent=latent_extent)
plt.title('Entropy over classes')
plt.subplot(133)
plt.imshow(entropy_image, extent=latent_extent)
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
