from models import gaussian_mnist
import os
import pygame
import numpy as np
import pygame.locals as pl
import keras.backend as K
from util import get_class_colors
from sys import argv

from my_mnist import x_test, y_test

pygame.init()

screen_width, screen_height = 800, 400
bgcolor = (0, 0, 0)
dest_rect = (410, 10, 380, 380)
main_surface = pygame.display.set_mode((screen_width, screen_height))
mnist_surface = pygame.Surface((28, 28))
np_mnist_surface_view = np.frombuffer(mnist_surface.get_buffer())
mnist_argb = np.zeros((28, 28, 4), dtype=np.uint8)
ent_surface = pygame.Surface((100, 100))
np_ent_surface_view = np.frombuffer(ent_surface.get_buffer())
class_argb = np.zeros((100, 100, 4), dtype=np.uint8)

game_exit = False
clock = pygame.time.Clock()
mouse_position = (0, 0)

model_type = argv[1]
latent_dim = 2
pixel_std = .05
vae = gaussian_mnist(model_type, latent_dim=latent_dim, pixel_std=pixel_std, k=1)
weights_file = os.path.join("models", "mnist_%s.h5" % model_type)
vae.model.load_weights(weights_file)

#####################
# GET ENTROPY IMAGE #
#####################

# Get color for each numeric class
class_colors = get_class_colors(10)

# Get keras function of recognition model.
q = K.function([vae.q_model.input], [vae.latent.mean, vae.latent.log_var])

# Apply recognition model to test set.
pred_mean, pred_log_var = q([x_test])
pred_var = np.exp(pred_log_var)
pred_std = np.sqrt(pred_var)

# Compute maximum extent of predictions (3 std dev) in x and y direction
extent = 5

# Create meshgrid of points covering the +/- maximum extent
xs = np.linspace(-extent, extent, 100)
ys = np.linspace(-extent, extent, 100)
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

class_argb[:, :, 0] = 255
class_argb[:, :, 1] = (entropy_image[:, :, 0] * 255).astype(np.int32)
class_argb[:, :, 2] = (entropy_image[:, :, 1] * 255).astype(np.int32)
class_argb[:, :, 3] = (entropy_image[:, :, 2] * 255).astype(np.int32)
np_ent_surface_view[...] = np.frombuffer(class_argb)


main_surface.fill(bgcolor)
main_surface.blit(pygame.transform.scale(ent_surface, (400, 400)), (0, 0, 400, 400))

######################
# CONSTRUCT RENDERER #
######################

render = K.function([vae.latent_sample], [vae.reconstruction])
renderer_input = np.zeros((1, 2))

####################
# INTERACTIVE LOOP #
####################

while not game_exit:
    for evt in pygame.event.get():
        if evt.type == pl.QUIT:
            game_exit = True
        elif evt.type == pl.MOUSEMOTION:
            mouse_position = evt.pos

    latent_x = extent * (mouse_position[0] - screen_width / 4) / (screen_width / 4)
    latent_y = extent * (mouse_position[1] - screen_height / 2) / (screen_height / 2)
    renderer_input[:] = (latent_x, latent_y)
    rendered_image = render([renderer_input])[0].reshape((28, 28))
    rendered_image = (rendered_image * 255).astype(np.uint8)
    mnist_argb[:, :, 0] = 255
    mnist_argb[:, :, 1] = rendered_image
    mnist_argb[:, :, 2] = rendered_image
    mnist_argb[:, :, 3] = rendered_image

    np_mnist_surface_view[...] = np.frombuffer(mnist_argb)
    main_surface.blit(pygame.transform.scale(mnist_surface, (380, 380)), dest_rect)

    pygame.display.update()
    clock.tick(30)

pygame.quit()
