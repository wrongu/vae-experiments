import os
import numpy as np


arms = 3
points = 10000
angle_noise = np.pi / arms / 10
radius_noise = .3
spiral_gain = 1

data = np.zeros((points, 2))
for i in range(points):
    angle = 2 * np.pi * float(np.random.randint(arms)) / arms + np.random.randn() * angle_noise
    radius = 1 + np.random.randn() * radius_noise
    angle += spiral_gain * radius
    data[i, :] = radius * np.array([np.sin(angle), np.cos(angle)])

np.save(os.path.join("numpy_data", "pinwheel.npy"), data)
