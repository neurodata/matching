#%%
# Simulation

from abc import ABC
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.simulations import er_corr
from pkg.io import FIG_PATH, OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.match import BisectedGraphMatchSolver, GraphMatchSolver
from pkg.plot import set_theme
from tqdm import tqdm

DISPLAY_FIGS = True

FILENAME = "simulations"

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
contra_rho = 0.6
# simulate the correlated subgraphs
n_sims = 1000
rows = []
for n_seeds in [0, 1, 2, 3, 4, 5]:
    for sim in tqdm(range(n_sims)):
        if n_seeds > 0:
            seeds = np.arange(n_seeds)
            partial_match = np.stack((seeds, seeds)).T
        else:
            partial_match = None
        A, B = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
        AB, BA = er_corr(n_side, contra_p, contra_rho, directed=True)
        solver = GraphMatchSolver(A, B, verbose=0, partial_match=partial_match)
        solver.solve()
        match_ratio = (solver.permutation_ == np.arange(len(A))).mean()
        rows.append({"n_seeds": n_seeds, "match_ratio": match_ratio})
results = pd.DataFrame(rows)


#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.lineplot(data=results, x="n_seeds", y="match_ratio", ax=ax)

#%%

rows = []
for contra_rho in np.linspace(0, 1, 11):
    for sim in tqdm(range(n_sims)):
        # simulate the correlated subgraphs
        A, B = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
        AB, BA = er_corr(n_side, contra_p, contra_rho, directed=True)

        # construct the full network
        # indices_A = np.arange(n_side)
        # indices_B = np.arange(n_side, 2 * n_side)
        # adjacency = np.zeros((2 * n_side, 2 * n_side))
        # adjacency[np.ix_(indices_A, indices_A)] = A
        # adjacency[np.ix_(indices_B, indices_B)] = B
        # adjacency[np.ix_(indices_A, indices_B)] = AB
        # adjacency[np.ix_(indices_B, indices_A)] = BA

        # permute one hemisphere
        # side_perm = rng.permutation(n_side) + n_side
        # perm = np.concatenate((indices_A, side_perm))
        # adjacency = adjacency[np.ix_(perm, perm)]
        # undo_perm =
        perm = rng.permutation(n_side)
        undo_perm = np.argsort(perm)
        B = B[perm][:, perm]
        AB = AB[:, perm]
        BA = BA[perm, :]

        # run the matching
        for method in ["GM", "BGM", "MGM"]:
            if method == "GM":
                solver = GraphMatchSolver(A, B)
            elif method == "BGM":
                solver = GraphMatchSolver(A, B, AB=AB, BA=BA)
            elif method == "MGM":
                solver = GraphMatchSolver([A, BA], [B, AB])
            solver.solve()
            match_ratio = (solver.permutation_ == undo_perm).mean()

            rows.append(
                {
                    "ipsi_rho": ipsi_rho,
                    "contra_rho": contra_rho,
                    "match_ratio": match_ratio,
                    "sim": sim,
                    "method": method,
                }
            )

results = pd.DataFrame(rows)


#%%

tab10_colorblind = sns.color_palette("colorblind")
method_palette = dict(
    zip(
        ["GM", "BGM", "MGM"],
        [tab10_colorblind[0], tab10_colorblind[1], tab10_colorblind[2]],
    )
)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(
    data=results,
    x="contra_rho",
    y="match_ratio",
    hue="method",
    style="method",
    hue_order=["GM", "BGM", "MGM"],
    # dashes={"GM": "--", "BGM": "-", "MGM": "-."},
    ax=ax,
    palette=method_palette,
)
ax.set_ylabel("Matching accuracy")
ax.set_xlabel("Contralateral edge correlation")
sns.move_legend(ax, loc="upper left", title="Method", frameon=True)
gluefig("match_ratio_by_contra_rho", fig)


#%%
# A, B = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
# AB, BA = er_corr(n_side, contra_p, contra_rho, directed=True)
# solver = BisectedGraphMatchSolver(A, B, AB=AB, BA=BA)
# solver.solve()
# match_ratio = (solver.permutation_ == undo_perm).mean()
# match_ratio
#%%
import graspologic as gl

gl.plot.heatmap(np.squeeze(solver.A))
gl.plot.heatmap(np.squeeze(A))

gl.plot.heatmap(np.squeeze(solver.B))
gl.plot.heatmap(np.squeeze(B))

#%%

mean = 0
contra_rho = ipsi_rho
for i in range(100):
    A, B = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
    C, D = er_corr(n_side, contra_p, contra_rho, directed=True)

    perm = rng.permutation(n_side)
    undo_perm = np.argsort(perm)
    B = B[perm][:, perm]
    D = D[perm][:, perm]

    solver = GraphMatchSolver([A, C], [B, D])
    solver.solve()
    match_ratio_multi = (solver.permutation_ == undo_perm).mean()

    solver = GraphMatchSolver([A], [B])
    solver.solve()
    match_ratio = (solver.permutation_ == undo_perm).mean()

    diff = match_ratio_multi - match_ratio
    mean += diff / 100

mean

#%%
from functools import wraps


def timer(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        sec = te - ts
        output = f"Function {f.__name__} took {sec:.3f} seconds."
        print(output)
        return result

    return wrap


class MyCount:
    def __init__(self, x):
        self.x = x

    @timer
    def count(self):
        print(self.x + 1)


mc = MyCount(4)
mc.count()
