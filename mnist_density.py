import os
import numpy as np
import matplotlib.pyplot as plt
from models import gaussian_mnist
import keras.backend as K

#################
# CREATE MODELS #
#################

model_type = 'vae'
latent_dim = 2
pixel_std = .05
k_samples = 8

vae = gaussian_mnist(model_type, latent_dim=latent_dim, pixel_std=pixel_std, k=k_samples)
weights_file = os.path.join("models", "mnist_%s.h5" % model_type)
vae.model.load_weights(weights_file)


#############################
# DENSITY-NET VISUALIZATION #
#############################

vae.set_samples(1)
render = K.function([vae.latent_sample], [vae.reconstruction])
pts = np.random.randn(1000, 2)
pixels = render([pts])[0]
# 583 360
# 489 431
i = 380  # np.random.randint(784)
j = 400  # np.random.randint(784)

print(i, j)

ax = plt.subplot(121)
plt.scatter(pts[:, 0], pts[:, 1], marker='.')
ax.set_aspect(1)

ax = plt.subplot(122)
plt.scatter(pixels[:, i], pixels[:, j], marker='.')
ax.set_aspect(1)

plt.show()
