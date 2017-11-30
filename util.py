import numpy as np
from matplotlib.colors import hsv_to_rgb


def get_class_colors(n_classes):
    """Get colormap from equi-spaced hues in HSV space, converted back to RGB.
    """
    hues = np.linspace(0, 1, n_classes + 1)[:-1]
    saturations = np.ones(n_classes)
    values = np.ones(n_classes)
    return hsv_to_rgb(np.vstack([hues, saturations, values]).T)


def class_categorical(means, stds, classes, res=100, eps=1e-10):
    # Compute maximum extent of predictions (3 std dev) in x and y direction
    max_extent_x, max_extent_y = np.max(np.abs(means) + 3 * stds, axis=0)
    max_extent_x, max_extent_y = [max(max_extent_x, max_extent_y)] * 2

    # Create meshgrid of points covering the +/- maximum extent
    xs = np.linspace(-max_extent_x, max_extent_x, res)
    ys = np.linspace(-max_extent_y, max_extent_y, res)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.concatenate([np.expand_dims(xx, 2), np.expand_dims(yy, 2)], 2)

    # Create a categorical distribution over classes at each point in the grid/image
    categorical = np.full(xx.shape + (classes.unique().size,), eps)

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

    # For each test point of class c, evaluate normpdf of the model and add to categorical[:,:,c]
    for c, mu, var in zip(classes, means, stds):
        categorical[:, :, c] += multivariate_gaussian(pts, mu, np.diag(var))

    # Normalize across classes at each point
    categorical = categorical / categorical.sum(axis=2)[:, :, np.newaxis]

    return categorical, (-max_extent_x, max_extent_x, -max_extent_y, max_extent_y)
