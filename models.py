from vae import VAE, IWAE, GaussianPrior, GaussianLatent, GaussianLikelihood, \
    CategoricalLatent, DiscreteUniformPrior
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Reshape, Flatten
import keras.backend as K


mnist_rows, mnist_cols, mnist_channels = 28, 28, 1
img_pixels = mnist_rows * mnist_cols

omniglot_rows, omniglot_cols, omniglot_channels = 64, 64, 1
omniglot_pixels = omniglot_rows * omniglot_cols


def gaussian_omniglot(cls='vae', latent_dim=2, pixel_std=.05, k=1):
    # number of convolutional filters to use
    filters = 64
    # convolution kernel size
    num_conv = 3
    intermediate_dim = 128

    # RECOGNITION MODEL
    if K.image_data_format() == 'channels_first':
        original_img_size = (omniglot_channels, omniglot_rows, omniglot_cols)
    else:
        original_img_size = (omniglot_rows, omniglot_cols, omniglot_channels)

    x = Input(shape=original_img_size)
    conv_1 = Conv2D(omniglot_channels,
                    kernel_size=(2, 2),
                    padding='same', activation='relu')(x)
    conv_2 = Conv2D(filters,
                    kernel_size=(2, 2),
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_1)
    conv_3 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_2)
    conv_4 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_3)
    flat = Flatten()(conv_4)
    intermediate = Dense(intermediate_dim, activation='relu')(flat)
    q_model = Model(x, intermediate)

    # LATENT
    latent = GaussianLatent(dims=latent_dim)

    # PRIOR
    prior = GaussianPrior(dims=latent_dim)

    # GENERATIVE MODEL
    p_model = Sequential()
    p_model.add(Dense(intermediate_dim, activation='relu', input_dim=latent_dim))
    p_model.add(Dense(filters * omniglot_rows // 2 * omniglot_cols // 2, activation='relu'))

    if K.image_data_format() == 'channels_first':
        output_shape = (filters, omniglot_rows // 2, omniglot_cols // 2)
    else:
        output_shape = (omniglot_rows // 2, omniglot_cols // 2, filters)

    p_model.add(Reshape(output_shape))
    p_model.add(Conv2DTranspose(filters,
                                kernel_size=num_conv,
                                padding='same',
                                strides=1,
                                activation='relu'))
    p_model.add(Conv2DTranspose(filters,
                                kernel_size=num_conv,
                                padding='same',
                                strides=1,
                                activation='relu'))
    p_model.add(Conv2DTranspose(filters,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                padding='valid',
                                activation='relu'))
    p_model.add(Conv2D(omniglot_channels,
                       kernel_size=2,
                       padding='valid',
                       activation='sigmoid'))

    # LIKELIHOOD
    likelihood = GaussianLikelihood(pixel_std)

    # Combine the above parts into a single model
    kwargs = {
        'q_model': q_model,
        'latent': latent,
        'prior': prior,
        'p_model': p_model,
        'likelihood': likelihood,
        'k': k
    }
    if cls == 'vae':
        return VAE(**kwargs)
    elif cls == 'iwae':
        return IWAE(**kwargs)


def gaussian_mnist(cls='vae', latent_dim=2, pixel_std=.05, k=1):
    # RECOGNITION MODEL
    q_model = Sequential()
    q_model.add(Dense(64, activation='relu', input_dim=img_pixels))
    q_model.add(Dense(64, activation='relu'))

    # LATENT
    latent = GaussianLatent(dims=latent_dim)

    # PRIOR
    prior = GaussianPrior(dims=latent_dim)

    # GENERATIVE MODEL
    p_model = Sequential()
    p_model.add(Dense(64, activation='relu', input_dim=latent_dim))
    p_model.add(Dense(64, activation='relu'))
    p_model.add(Dense(img_pixels, activation='sigmoid'))

    # LIKELIHOOD
    likelihood = GaussianLikelihood(pixel_std)

    # Combine the above parts into a single model
    kwargs = {
        'q_model': q_model,
        'latent': latent,
        'prior': prior,
        'p_model': p_model,
        'likelihood': likelihood,
        'k': k
    }
    if cls == 'vae':
        return VAE(**kwargs)
    elif cls == 'iwae':
        return IWAE(**kwargs)


def mnist_hvae(cls='vae', latent_dim=2, layers=3, pixel_std=.05, k=1):
    # RECOGNITION MODEL(s)
    q_model = Sequential()
    q_model.add(Dense(64, activation='relu', input_dim=img_pixels))
    q_model.add(Dense(64, activation='relu'))

    # LATENT
    latent = GaussianLatent(dims=latent_dim)

    # PRIOR
    prior = GaussianPrior(dims=latent_dim)

    # GENERATIVE MODEL
    p_model = Sequential()
    p_model.add(Dense(64, activation='relu', input_dim=latent_dim))
    p_model.add(Dense(64, activation='relu'))
    p_model.add(Dense(img_pixels, activation='sigmoid'))

    # LIKELIHOOD
    likelihood = GaussianLikelihood(pixel_std)

    # Combine the above parts into a single model
    kwargs = {
        'q_model': q_model,
        'latent': latent,
        'prior': prior,
        'p_model': p_model,
        'likelihood': likelihood,
        'k': k
    }
    if cls == 'vae':
        return VAE(**kwargs)
    elif cls == 'iwae':
        return IWAE(**kwargs)


def categorical_mnist(cls='vae', latent_dim=10, pixel_std=.05, k=1):
    # RECOGNITION MODEL
    q_model = Sequential()
    q_model.add(Dense(64, activation='relu', input_dim=img_pixels))
    q_model.add(Dense(64, activation='relu'))

    # LATENT
    latent = CategoricalLatent(dims=latent_dim)

    # PRIOR
    prior = DiscreteUniformPrior(dims=latent_dim)

    # GENERATIVE MODEL
    p_model = Sequential()
    p_model.add(Dense(64, activation='relu', input_dim=latent_dim))
    p_model.add(Dense(64, activation='relu'))
    p_model.add(Dense(img_pixels, activation='sigmoid'))

    # LIKELIHOOD
    likelihood = GaussianLikelihood(pixel_std)

    # Combine the above parts into a single model
    kwargs = {
        'q_model': q_model,
        'latent': latent,
        'prior': prior,
        'p_model': p_model,
        'likelihood': likelihood,
        'k': k
    }
    if cls == 'vae':
        return VAE(**kwargs)
    elif cls == 'iwae':
        return IWAE(**kwargs)
