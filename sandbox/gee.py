#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.plot import heatmap
from graspologic.simulations import sbm
from graspologic.utils import is_almost_symmetric
from numba import njit
from pkg.plot import set_theme
from sklearn.base import BaseEstimator


@njit
def _project_edges_numba(sources, targets, weights, W):
    n = W.shape[0]
    k = W.shape[1]
    Z = np.zeros((n, k))
    # TODO redo with broadcasting
    for source, target, weight in zip(sources, targets, weights):
        Z[source] += W[target] * weight
        Z[target] += W[source] * weight
    return Z


def _project_edges_broadcast(sources, targets, weights, W):
    n = W.shape[0]
    k = W.shape[1]
    Z = np.zeros((n, k))
    # this is broken, how to index into the positions and sum?
    Z[sources] += W[targets] * weights[:, None]
    Z[targets] += W[sources] * weights[:, None]
    return Z




class GraphEncoderEmbedding(BaseEstimator):
    def __init__(self, laplacian=False) -> None:
        self.laplacian = laplacian
        super().__init__()

    def fit(self, adjacency, features):
        sources, targets = np.nonzero(adjacency)

        # handle the undireced case
        # if undirected, we only need to iterate over the upper triangle of adjacency
        if is_almost_symmetric(adjacency):
            mask = sources <= targets  # includes the diagonal
            sources = sources[mask]
            targets = targets[mask]

        weights = adjacency[sources, targets]

        if self.laplacian:
            # TODO implement regularized laplacian
            degrees_out = np.sum(adjacency, axis=1)
            degrees_in = np.sum(adjacency, axis=0)
            degrees_out_root = 1 / np.sqrt(degrees_out)
            degrees_in_root = 1 / np.sqrt(degrees_in)

            weights *= degrees_out_root[sources] * degrees_in_root[targets]

        Y_colsum = np.sum(features, axis=0)
        W = features / Y_colsum[None, :]

        Z = _project_edges_numba(sources, targets, weights, W)
        # Z = _project_edges_broadcast(sources, targets, weights, W)

        self.embedding_ = Z
        self.projection_matrix_ = W
        return self

    def fit_transform(self, adjacency, features):
        self.fit(adjacency, features)
        return self.embedding_

    def transform(self, adjacency, features):
        msg = "Out-of-sample embedding has not been implemented yet."
        raise NotImplementedError(msg)


Z = np.zeros((10, 2))
Z[[1, 1, 1]] += np.array([1, 1])
Z

#%%
from graspologic.simulations import er_np

adjacency = er_np(10, 0.1)
sources, targets = np.nonzero(adjacency)
weights = adjacency[sources, targets]
Y = np.random.uniform(0, 1, size=(10, 2))

Z_numba = _project_edges_numba(sources, targets, weights, Y)

Z_einsum = np.einsum("", ) * weights


#%%

rng = np.random.default_rng(8888)

ns = [1000, 1000]
B = np.array([[0.13, 0.1], [0.1, 0.13]])
B_dc = np.array([[0.9, 0.1], [0.1, 0.5]])


def sample_data(model):
    labels = np.zeros((2000,), dtype=int)
    labels[ns[0] :] = 1
    Y = np.zeros((np.sum(ns), 2))
    Y[labels == 0, 0] = 1
    Y[labels == 1, 1] = 1
    if model == 1:
        A = sbm(ns, B, return_labels=False)
    elif model == 2:
        theta = np.random.beta(1, 4, size=(sum(ns)))
        P = np.full((2000, 2000), 0.1)
        P[: ns[0], : ns[0]] = 0.9
        P[ns[0] :, ns[0] :] = 0.5
        P = P * theta[:, None] * theta[None, :]
        P = P - np.tril(P)
        A = rng.binomial(1, P)
        A = A + A.T
    elif model == 3:
        X1 = np.random.beta(1, 5, size=(ns[0], 1))
        X2 = np.random.beta(5, 1, size=(ns[1], 1))
        X = np.concatenate([X1, X2], axis=0)
        P = X @ X.T
        P = P - np.tril(P)
        A = rng.binomial(1, P)
        A = A + A.T
    return A, Y, labels


fig, axs = plt.subplots(3, 3, figsize=(15, 15))


set_theme(font_scale=1.5)

model_names = ["SBM", "DCSBM", "RDPG"]

for j, model in enumerate(range(1, 4)):
    A, Y, labels = sample_data(model)

    gee = GraphEncoderEmbedding()
    Z = gee.fit_transform(A, Y)

    ase = AdjacencySpectralEmbed(n_components=2, check_lcc=False)
    X = ase.fit_transform(A)

    ax = axs[0, j]
    heatmap(A, ax=ax, cbar=False)
    ax.set_title(model_names[j])

    if j == 0:
        ax.set_ylabel("Adjacency matrix")

    scatter_kws = dict(hue=labels, legend=False, s=10, alpha=0.5)

    ax = axs[1, j]
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], ax=ax, **scatter_kws)

    if j == 0:
        ax.set_ylabel("Encoder embedding")

    ax = axs[2, j]
    sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax, **scatter_kws)

    if j == 0:
        ax.set_ylabel("Spectral embedding")


# %%
