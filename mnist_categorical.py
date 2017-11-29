import os
import numpy as np
import matplotlib.pyplot as plt
from models import categorical_mnist
import keras.backend as K

###################
# LOAD MNIST DATA #
###################

from my_mnist import x_train, x_test, y_test

img_rows, img_cols = 28, 28
img_pixels = img_rows * img_cols

#################
# CREATE MODELS #
#################

model_type = 'iwae'
latent_dim = 10
vae = categorical_mnist(model_type, latent_dim=latent_dim, pixel_std=.05, k=16)
vae.model.compile(loss=None, optimizer='rmsprop')

###########
# FITTING #
###########

# Load pre-trained weights if they exist
weights_file = os.path.join("models", "mnist_categorical_%s.h5" % model_type)
if os.path.exists(weights_file):
    vae.model.load_weights(weights_file)
else:
    vae.model.fit(x_train,
                  shuffle=True,
                  epochs=100,
                  batch_size=100,
                  validation_data=(x_test, None))
    # Save trained model to a file
    vae.model.save_weights(weights_file)

#################
# Visualization #
#################

render = K.function([vae.latent_sample], [vae.reconstruction])
full_image = np.zeros((img_rows, img_cols * 11))
inputs = np.eye(latent_dim)
outputs = render([inputs])[0]
for i in range(latent_dim):
    full_image[:, i * img_rows:(i + 1) * img_rows] = np.reshape(outputs[i], (img_rows, img_cols))
# Plot average training image for comparison
full_image[:, latent_dim * img_rows:11 * img_rows] = \
    np.reshape(np.mean(x_train, axis=0), (img_rows, img_cols))
plt.imshow(full_image, cmap='Greys_r')
plt.show()
