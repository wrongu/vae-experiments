from scipy.stats import multivariate_normal
import numpy as np
import os

dim = 2
clusters = 5
points = 5000

means = np.random.randn(clusters, dim) * 5
covariances = np.zeros((clusters, dim, dim))
pis = np.random.rand(clusters) + 1.0 / clusters
pis = pis / pis.sum()
components = [None] * clusters
data = np.zeros((points, dim))
labels = np.zeros((points,), dtype=np.int32)

for i in range(clusters):
    factors = np.random.randn(dim, dim)
    covariances[i, :, :] = np.eye(dim) + factors.T * factors
    components[i] = multivariate_normal(mean=means[i, :],
                                        cov=covariances[i, ...].reshape((dim, dim)))

for i in range(points):
    labels[i] = np.random.choice(range(clusters))
    data[i, :] = components[labels[i]].rvs()

np.savez(os.path.join("numpy_data", "gmm.npz"),
         data=data.astype(np.float32),
         labels=labels.astype(np.float32),
         means=means.astype(np.float32),
         covariances=covariances.astype(np.float32),
         pis=pis.astype(np.float32))
