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

A = np.zeros((5, 5))
B = np.zeros((4, 4))
solver = GraphMatchSolver(A, B, use_numba=False)
solver.solve()


# print(solver.permutation_)

#%%
from graspologic.simulations import er_np

extras = np.arange(0, 5)

rows = []
with tqdm(total=len(extras) * 2 * n_sims) as pbar:
    for padding in ["naive", "adopted"]:
        for extra in extras:
            for i in range(n_sims):
                pbar.update(1)

                A, B = er_corr(n_side, 0.3, 0.9, directed=True)
                B_big = er_np(n_side + extra, 0.3, directed=True)
                B_big[:n_side, :n_side] = B

                perm = np.random.permutation(len(B_big))
                undo_perm = np.argsort(perm)
                B_big = B_big[perm][:, perm]

                solver = GraphMatchSolver(A, B_big, use_numba=False, padding=padding)
                solver.solve()
                matching = solver.matching_
                acc = (undo_perm[matching[:, 0]] == matching[:, 1]).mean()
                rows.append(
                    {
                        "match_ratio": acc,
                        "extra": extra,
                        "padding": padding,
                        "method": "new",
                    }
                )

                fit_perm = GraphMatch(padding=padding, shuffle_input=False).fit_predict(
                    A, B_big
                )
                acc = (undo_perm[:n_side] == fit_perm[:n_side]).mean()
                rows.append(
                    {
                        "match_ratio": acc,
                        "extra": extra,
                        "padding": padding,
                        "method": "old",
                    }
                )

results = pd.DataFrame(rows)
#%%
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

ax = axs[0]
sns.lineplot(
    data=results[results["padding"] == "naive"],
    x="extra",
    y="match_ratio",
    hue="method",
    ax=ax,
)
ax.set(title="padding = naive")

ax = axs[1]
sns.lineplot(
    data=results[results["padding"] == "adopted"],
    x="extra",
    y="match_ratio",
    hue="method",
    ax=ax,
)
ax.set(title="padding = adopted")

# %%
