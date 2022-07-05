#%% [markdown]
# # Benchmarks

#%%
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.match import GraphMatchSolver  # experimental version
from graspologic.match import GraphMatch
from graspologic.match import graph_match
from graspologic.simulations import er_corr
from pkg.io import FIG_PATH, OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from tqdm.autonotebook import tqdm

DISPLAY_FIGS = True

FILENAME = "benchmarks"

OUT_PATH = OUT_PATH / FILENAME

FIG_PATH = FIG_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)


#%%

A = [
    [0, 90, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [90, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0],
    [10, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0],
    [0, 23, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0],
    [0, 0, 43, 0, 0, 0, 26, 0, 0, 0, 0, 0],
    [0, 0, 0, 88, 0, 0, 0, 16, 0, 0, 0, 0],
    [0, 0, 0, 0, 26, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 16, 0, 0, 0, 96, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 29, 0],
    [0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 37],
    [0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0],
]
B = [
    [0, 36, 54, 26, 59, 72, 9, 34, 79, 17, 46, 95],
    [36, 0, 73, 35, 90, 58, 30, 78, 35, 44, 79, 36],
    [54, 73, 0, 21, 10, 97, 58, 66, 69, 61, 54, 63],
    [26, 35, 21, 0, 93, 12, 46, 40, 37, 48, 68, 85],
    [59, 90, 10, 93, 0, 64, 5, 29, 76, 16, 5, 76],
    [72, 58, 97, 12, 64, 0, 96, 55, 38, 54, 0, 34],
    [9, 30, 58, 46, 5, 96, 0, 83, 35, 11, 56, 37],
    [34, 78, 66, 40, 29, 55, 83, 0, 44, 12, 15, 80],
    [79, 35, 69, 37, 76, 38, 35, 44, 0, 64, 39, 33],
    [17, 44, 61, 48, 16, 54, 11, 12, 64, 0, 70, 86],
    [46, 79, 54, 68, 5, 0, 56, 15, 39, 70, 0, 18],
    [95, 36, 63, 85, 76, 34, 37, 80, 33, 86, 18, 0],
]
A, B = np.array(A), np.array(B)
n = A.shape[0]
pi = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - [1] * n

#%%
_, _, score, _ = graph_match(
    A, B, n_init=50, maximize=False, init_perturbation=0.5, rng=888
)
score

#%%
from graspologic.simulations import er_np
np.random.seed(888)
n = 50
p = 0.4

A = er_np(n=n, p=p)
B = A[:-2, :-2]  # remove two nodes

indices_A, indices_B, _, _ = graph_match(A, B, rng=888, padding="adopted")
np.array_equal(indices_A, np.arange(n-2))
#%%
gm = GraphMatch(init="rand", gmp=False)
gm.fit_predict(A, B)
gm.score_

#%%

seeds1 = np.array(range(n))
seeds2 = pi
partial_match = np.column_stack((seeds1, seeds2))
_, indices_B, score, _ = graph_match(A, B, partial_match=partial_match, maximize=False)
np.testing.assert_array_equal(indices_B, pi)
score
#%%


#%%
seeds1 = [4, 8, 10]
seeds2 = [pi[z] for z in seeds1]
partial_match = np.column_stack((seeds1, seeds2))
indices_A, indices_B, score, misc_df = graph_match(
    A,
    B,
    partial_match=partial_match,
    maximize=False,
    n_init=1,
    init_perturbation=0,
    shuffle_input=False,
)
# print(indices_A)

print(pi)
print()
print(indices_B)
print((indices_B == pi).mean())
print(score)
raw_score = np.sum(A * B[indices_B][:, indices_B])
print(raw_score)

from graspologic.match import GraphMatch

gm = GraphMatch(gmp=False, shuffle_input=False)
pred_inds = gm.fit_predict(A, B, seeds1, seeds2)

print()
print(pred_inds)
print((pred_inds == pi).mean())
print(gm.score_)

#%%
gm.score_ * 2 + np.linalg.norm(A) ** 2 + np.linalg.norm(B) ** 2

#%%

(
    2 * np.linalg.norm(A - B[indices_B][:, indices_B]) ** 2
    - np.linalg.norm(A) ** 2
    - np.linalg.norm(B) ** 2
)

#%%
n_side = 10
glue("n_side", n_side)
n_sims = 1000
glue("n_sims", n_sims, form="long")
ipsi_rho = 0.8
glue("ipsi_rho", ipsi_rho)
ipsi_p = 0.3
glue("ipsi_p", ipsi_p)
contra_p = 0.2
glue("contra_p", contra_p)

#%%
from graspologic.simulations import er_np

n_side = 100
extra = 2
A, B = er_corr(n_side, 0.3, 0.8, directed=True)
B_big = er_np(n_side + extra, 0.3, directed=True)
B_big[:n_side, :n_side] = B

perm = np.random.permutation(len(B_big))
undo_perm = np.argsort(perm)
B_big = B_big[perm][:, perm]

dfs = []
for init_perturbation in np.linspace(0, 1, 5):
    indices_A, indices_B, score, misc_df = graph_match(
        A, B, n_init=10, init_perturbation=init_perturbation
    )
    misc_df["init_perturbation"] = init_perturbation
    dfs.append(misc_df)
total_misc = pd.concat(dfs)


#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.stripplot(data=total_misc, x="init_perturbation", y="score", ax=ax)
