#%%

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from graspologic.match import graph_match
from pkg.io import FIG_PATH, OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme

from tqdm.autonotebook import tqdm
from git import Repo
from pkg.data import load_matched
from pkg.io import FIG_PATH, OUT_PATH
import seaborn as sns
from graspologic.match import graph_match
import networkx as nx
from pathlib import Path

DISPLAY_FIGS = True

FILENAME = "profile_maggot"

OUT_PATH = OUT_PATH / FILENAME

FIG_PATH = FIG_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


graspologic_path = "/Users/bpedigo/JHU_code/neuro-matching/graspologic"
graspologic_repo = Repo(graspologic_path)
print("Using new graph matching implementation under development, see PR:")
print("https://github.com/microsoft/graspologic/pull/960")
print(f"Current graspologic branch: {graspologic_repo.active_branch.name}")
print(f"Current graspologic commit hash: {graspologic_repo.head.commit}")

t0 = time.time()
set_theme()
rng = np.random.default_rng(888)
np.random.seed(8888)

left_adj, left_meta = load_matched(side="left", weights=True)
right_adj, right_meta = load_matched(side="right", weights=True)

#%%
A = left_adj
B = right_adj
_, _, _, misc = graph_match(A, B, max_iter=1)
P = misc[0]["convex_solution"]
_, perm, _, misc = graph_match(A, B, max_iter=2)
Q = misc[0]["convex_solution"]

n_seeds = 100

P = P[n_seeds:, n_seeds:]
Q = Q[n_seeds:, n_seeds:]
# perm = perm[n_seeds:]
# perm = np.random.permutation(P.shape[0])
from scipy.optimize import linear_sum_assignment

_, perm = linear_sum_assignment(Q, maximize=True)
Q = np.eye(Q.shape[0])[perm]


def _split_matrix(X: np.ndarray, n: int):
    # definitions according to Seeded Graph Matching [2].
    upper, lower = X[:n], X[n:]
    return upper[:, :n], upper[:, n:], lower[:, :n], lower[:, n:]


_, A_sn, A_ns, A = _split_matrix(A, n_seeds)
_, B_sn, B_ns, B = _split_matrix(B, n_seeds)

A12 = A_sn
A21 = A_ns
A22 = A

B12 = B_sn
B21 = B_ns
B22 = B

#%%


def old():
    R = P - Q
    b21 = ((R.T @ A21) * B21).sum()
    b12 = ((R.T @ A12.T) * B12.T).sum()
    AR22 = A22.T @ R
    BR22 = B22 @ R.T
    b22a = (AR22 * B22.T[perm]).sum()
    b22b = (A22 * BR22[perm]).sum()
    a = (AR22.T * BR22).sum()
    b = b21 + b12 + b22a + b22b
    return a, b


def new_dumb():
    a_intra = 0
    b_intra = 0
    R = P - Q
    a_intra += np.trace(A @ R @ B.T @ R.T)
    b_intra += np.trace(A @ Q @ B.T @ R.T) + np.trace(A @ R @ B.T @ Q.T)
    b_intra += np.trace(A_ns.T @ R @ B_ns) + np.trace(A_sn @ R @ B_sn.T)
    return a_intra, b_intra


print(old())
print(new_dumb())
#%%
%timeit -r 50 old

#%%
%timeit -r 50 new_dumb

#%%