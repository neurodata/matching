#%%
from graspologic.simulations.simulations import sbm
import numpy as np
from scipy.sparse import csr_array
from sklearn.base import BaseEstimator


class GraphEncoderEmbedding(BaseEstimator):
    def __init__(self, laplacian=False) -> None:
        self.laplacian = laplacian
        super().__init__()

    def fit(self, X, Y):
        n = X.shape[0]
        sources, targets = np.nonzero(X)
        weights = X[sources, targets]
        # E = np.column_stack((sources, targets, weights))

        if self.laplacian:
            degrees_out = np.sum(X, axis=1)
            degrees_in = np.sum(X, axis=0)
            degrees_out_root = 1 / np.sqrt(degrees_out)
            degrees_in_root = 1 / np.sqrt(degrees_in)

            weights *= degrees_out_root[sources] * degrees_in_root[targets]

        Y_colsum = np.sum(Y, axis=0)
        k = Y.shape[1]
        W = Y / Y_colsum[None, :]

        Z = np.zeros((n, k))
        for source, target, weight in zip(sources, targets, weights):
            Z[source] += W[source] * weight
            Z[target] += W[target] * weight

        # Z[sources] = W[sources] * weights

        self.Z_ = Z
        self.W_ = W
        return self


from graspologic.simulations import er_np

rng = np.random.default_rng()
n = 10
A = er_np(n, 0.5)
Y = np.random.uniform(0, 1, size=(n, 2))
gee = GraphEncoderEmbedding()
gee.fit(A, Y)
gee.Z_

#%%
ns = [1000, 1000]
# B = np.array([[0.9, 0.1], [0.1, 0.5]])
B = np.array([[0.13, 0.1], [0.1, 0.13]])
theta = np.random.beta(1, 4, size=(n))
from graspologic.simulations import sbm

A, labels = sbm(ns, B, return_labels=True)
Y = np.zeros((np.sum(ns), 2))
# labels = np.random.choice(2, size=(np.sum(ns)))
Y[labels == 0, 0] = 1
Y[labels == 1, 1] = 1
# Y = np.random.uniform(0, 1, size=(np.sum(ns), 2))
import matplotlib.pyplot as plt
import seaborn as sns

gee = GraphEncoderEmbedding()
gee.fit(A, Y)
Z = gee.Z_
W = gee.W_
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels, ax=ax)
