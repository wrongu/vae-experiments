import numpy as np
from matplotlib.colors import hsv_to_rgb


def get_class_colors(n_classes):
    """Get colormap from equi-spaced hues in HSV space, converted back to RGB.
    """
    hues = np.linspace(0, 1, n_classes + 1)[:-1]
    saturations = np.ones(n_classes)
    values = np.ones(n_classes)
    return hsv_to_rgb(np.vstack([hues, saturations, values]).T)
