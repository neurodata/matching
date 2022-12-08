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
from graspologic.types import AdjacencyMatrix, Tuple


@njit
def _project_edges_numba(
    sources: np.ndarray, targets: np.ndarray, weights: np.ndarray, W: np.ndarray
) -> np.ndarray:
    n = W.shape[0]
    k = W.shape[1]
    Z = np.zeros((n, k))
    # TODO redo with broadcasting/einsum?
    for source, target, weight in zip(sources, targets, weights):
        Z[source] += W[target] * weight
        Z[target] += W[source] * weight
    return Z


def _get_edges(adjacency: AdjacencyMatrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources, targets = np.nonzero(adjacency)

    # handle the undireced case
    # if undirected, we only need to iterate over the upper triangle of adjacency
    if is_almost_symmetric(adjacency):
        mask = sources <= targets  # includes the diagonal
        sources = sources[mask]
        targets = targets[mask]

    weights = adjacency[sources, targets]

    return sources, targets, weights


def _scale_weights(
    adjacency: AdjacencyMatrix,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    # TODO implement regularized laplacian
    degrees_out = np.sum(adjacency, axis=1)
    degrees_in = np.sum(adjacency, axis=0)
    degrees_out_root = 1 / np.sqrt(degrees_out)
    degrees_in_root = 1 / np.sqrt(degrees_in)

    weights *= degrees_out_root[sources] * degrees_in_root[targets]
    return weights


def _initialize_projection(features: np.ndarray) -> np.ndarray:
    features_colsum = np.sum(features, axis=0)
    W = features / features_colsum[None, :]
    return W


class GraphEncoderEmbedding(BaseEstimator):
    def __init__(self, laplacian: bool = False) -> None:
        """Implements the Graph Encoder Embedding of [1]_.

        More documentation coming soon.

        Parameters
        ----------
        laplacian : bool, optional
            Whether to normalize the embedding by the degree of the input and output
            nodes, by default False

        References
        ----------
        .. [1] C. Shen, Q. Wang, and C. Priebe, "One-Hot Graph Encoder Embedding,"
            arXiv:2109.13098 (2021).
        """
        self.laplacian = laplacian
        super().__init__()

    def fit(
        self, adjacency: AdjacencyMatrix, features: np.ndarray
    ) -> "GraphEncoderEmbedding":
        """Fit the embedding model to the input data.

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            n x n adjacency matrix of the graph
        features : np.ndarray
            n x k matrix of node features

        Returns
        -------
        GraphEncoderEmbedding
            The fitted embedding model
        """
        sources, targets, weights = _get_edges(adjacency)

        if self.laplacian:
            weights = _scale_weights(adjacency, sources, targets, weights)

        W = _initialize_projection(features)

        Z = _project_edges_numba(sources, targets, weights, W)

        self.embedding_ = Z
        self.projection_ = W

        return self

    def fit_transform(
        self, adjacency: AdjacencyMatrix, features: np.ndarray
    ) -> np.ndarray:
        """Fit the model to the input data and return the embedding.

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            n x n adjacency matrix of the graph
        features : np.ndarray
            n x k matrix of node features

        Returns
        -------
        np.ndarray
            The n x k embedding of the input graph
        """
        self.fit(adjacency, features)
        return self.embedding_

    def transform(self, adjacency: AdjacencyMatrix) -> np.ndarray:
        """Transform the input adjacency matrix into the embedding space.

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            n x n adjacency matrix of the graph

        Returns
        -------
        np.ndarray
            The n x k embedding of the input graph
        """
        sources, targets, weights = _get_edges(adjacency)

        if self.laplacian:
            weights = _scale_weights(adjacency, sources, targets, weights)

        Z = _project_edges_numba(sources, targets, weights, self.projection_)

        return Z


#%%

rng = np.random.default_rng(8888)


def sample_data(model, n=2000):
    if n % 2 != 0:
        raise ValueError("n must be even")
    labels = np.zeros((n,), dtype=int)
    ns = [n // 2, n // 2]
    labels[ns[0] :] = 1
    Y = np.zeros((np.sum(ns), 2))
    Y[labels == 0, 0] = 1
    Y[labels == 1, 1] = 1
    if model == "SBM":
        B = np.array([[0.13, 0.1], [0.1, 0.13]])
        A = sbm(ns, B, return_labels=False)
    elif model == "DCSBM":
        theta = rng.beta(1, 4, size=(sum(ns)))
        P = np.full((n, n), 0.1)
        P[: ns[0], : ns[0]] = 0.9
        P[ns[0] :, ns[0] :] = 0.5
        P = P * theta[:, None] * theta[None, :]
        P = P - np.tril(P)
        A = rng.binomial(1, P)
        A = A + A.T
    elif model == "RDPG":
        X1 = rng.beta(1, 5, size=(ns[0], 1))
        X2 = rng.beta(5, 1, size=(ns[1], 1))
        X = np.concatenate([X1, X2], axis=0)
        P = X @ X.T
        P = P - np.tril(P)
        A = rng.binomial(1, P)
        A = A + A.T
    else:
        raise ValueError("Model must be one of 'SBM', 'DCSBM', or 'RDPG'")
    return A, Y, labels


fig, axs = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)


set_theme(font_scale=1.5)

models = ["SBM", "DCSBM", "RDPG"]

for j, model in enumerate(models):
    A, Y, labels = sample_data(model)

    gee = GraphEncoderEmbedding()
    Z = gee.fit_transform(A, Y)

    ase = AdjacencySpectralEmbed(n_components=2, check_lcc=False)
    X = ase.fit_transform(A)

    ax = axs[0, j]
    heatmap(A, ax=ax, cbar=False)
    ax.set_title(model)

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

#%%
import time
from tqdm.autonotebook import tqdm

rows = []
n_sims = 10
n_range = np.geomspace(10, 10000, 8, dtype=int)
pbar = tqdm(total=3 * n_sims * len(n_range))
for n in n_range:
    n = n - n % 2
    for model in models:
        for sim in range(n_sims):
            A, Y, labels = sample_data(model, n=n)
            t0 = time.time()
            gee = GraphEncoderEmbedding()
            gee.fit_transform(A, Y)
            elapsed = time.time() - t0
            rows.append({"model": model, "sim": sim, "time": elapsed, "n": n})
            pbar.update(1)

#%%
import pandas as pd

results = pd.DataFrame(rows)
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, constrained_layout=True)
for i, model in enumerate(models):
    model_results = results.query("model == @model")
    ax = axs[i]
    sns.lineplot(x="n", y="time", data=model_results, ax=ax)
    ax.set_xlim(100, 2000)
    ax.set_ylim(0, 1)
