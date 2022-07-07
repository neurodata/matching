#%%

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.match import graph_match  # experimental version
from graspologic.match import GraphMatch
from graspologic.simulations import er_corr
from pkg.io import FIG_PATH, OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from tqdm.autonotebook import tqdm

DISPLAY_FIGS = True

FILENAME = "timing"

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
ns = np.geomspace(10, 1000, 4, dtype=int)
ns
rho = 0.9
from scipy.sparse import csr_array


def match_experiment(A, B, n_seeds=0, implementation="new", sparse=False):
    n = B.shape[0]
    perm = rng.permutation(n)
    undo_perm = np.argsort(perm)
    B = B[perm][:, perm]

    if n_seeds > 0:
        seeds_A = np.random.choice(n, replace=False, size=n_seeds)
        seeds_B = np.argsort(perm)[seeds_A]
        partial_match = np.stack((seeds_A, seeds_B)).T
    else:
        seeds_A = seeds_B = []
        partial_match = None
    non_seeds_A = np.setdiff1d(np.arange(n), seeds_A)

    if sparse:
        A, B = csr_array(A), csr_array(B)

    currtime = time.time()
    if implementation == "new":
        _, permutation, _, _ = graph_match(
            A, B, partial_match=partial_match, use_numba=False
        )
    else:
        permutation = GraphMatch().fit_predict(A, B, seeds_A=seeds_A, seeds_B=seeds_B)
    elapsed = time.time() - currtime

    match_ratio_full = (permutation == undo_perm).mean()
    match_ratio_nonseed = (permutation[non_seeds_A] == undo_perm[non_seeds_A]).mean()

    result = {}
    result["n"] = B.shape[0]
    result["implementation"] = implementation
    result["match_ratio_full"] = match_ratio_full
    result["match_ratio_nonseed"] = match_ratio_nonseed
    result["sparse"] = sparse
    result["time"] = elapsed
    result["n_seeds"] = n_seeds
    return result


n_seeds = 0
n_sims = 5
rows = []
from tqdm.autonotebook import tqdm

with tqdm(total=len(ns) * 3 * n_sims) as pbar:

    for n in ns:
        for _ in range(n_sims):
            p = np.log(n) / n
            A, B = er_corr(n, p, r=rho)

            for implementation in ["new", "old"]:
                if implementation == "new":
                    sparses = [False, True]
                else:
                    sparses = [False]
                for sparse in sparses:
                    pbar.update(1)
                    result = match_experiment(
                        A,
                        B,
                        n_seeds=n_seeds,
                        sparse=sparse,
                        implementation=implementation,
                    )
                    rows.append(result)

results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.lineplot(data=results, x="n", y="time", hue="implementation", style="sparse", ax=ax)

ax.set_yscale("log")

#%%

gluefig("timing", fig)
