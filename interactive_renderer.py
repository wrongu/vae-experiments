from models import gaussian_mnist
import os
import pygame
import numpy as np
import pygame.locals as pl
import keras.backend as K
from util import get_class_colors, class_categorical
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
q = K.function([vae.inpt], [vae.latents[0].mean, vae.latents[0].log_var])

# Apply recognition model to test set.
pred_mean, pred_log_var = q([x_test])
pred_var = np.exp(pred_log_var)
pred_std = np.sqrt(pred_var)

# Evaluate distribution over a set of points in the latent space -- get a categorical distribution
# over classes at each point
categorical, latent_extent = class_categorical(pred_mean, pred_std, y_test, res=100)

# Find what class is most (plurality) represented at each point
max_class = categorical.argmax(axis=2)

# Translate from max_class into a color at each point
class_image = class_colors[max_class]

# Get entropy across each categorical (scaled to be in [0, 1])
relative_entropy = -np.sum(categorical * np.log(categorical), axis=2) / np.log(10)
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

render = K.function([vae.latents[0].flat_samples], [vae.reconstruction])
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

    latent_x = latent_extent[1] * (mouse_position[0] - screen_width / 4) / (screen_width / 4)
    latent_y = latent_extent[3] * (mouse_position[1] - screen_height / 2) / (screen_height / 2)
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
