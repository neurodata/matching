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


#%% [markdown]
# ## Seeds
# This simulation adds an increasing number of "seeds," which can be thought of as a
# known partial matching.
#
# Matching accuracy is computed with respect to the *unseeded* vertices for this
# simulation.
#%%

seeds_range = np.arange(6)
lamb_range = np.linspace(0, 1, 6)

n_sims = 1000
rows = []
with tqdm(total=len(seeds_range) * len(lamb_range) * n_sims) as pbar:
    for n_seeds in seeds_range:
        for lamb in lamb_range:
            for sim in range(n_sims):
                pbar.update(1)

                A, B = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
                perm = rng.permutation(n_side)
                undo_perm = np.argsort(perm)
                B = B[perm][:, perm]

                if n_seeds > 0:
                    seeds_A = np.random.choice(n_side, replace=False, size=n_seeds)
                    # seeds_A = np.arange(n_seeds)
                    seeds_B = np.argsort(perm)[seeds_A]
                    partial_match = np.stack((seeds_A, seeds_B)).T
                else:
                    seeds_A = seeds_B = []
                    partial_match = None

                S = lamb * np.eye(B.shape[0])
                S = np.random.uniform(0, 1, (n_side, n_side)) + S
                S = S[:, perm]

                solver = GraphMatchSolver(
                    A,
                    B,
                    S=S,
                    partial_match=partial_match,
                    use_numba=False,
                )
                solver.solve()
                match_ratio = (solver.permutation_ == undo_perm).mean()
                rows.append(
                    {
                        "n_seeds": n_seeds,
                        "match_ratio": match_ratio,
                        "method": "new",
                        "lambda": lamb,
                    }
                )

                fit_perm = GraphMatch().fit_predict(
                    A, B, S=S, seeds_A=seeds_A, seeds_B=seeds_B
                )
                match_ratio = (fit_perm == undo_perm).mean()
                rows.append(
                    {
                        "n_seeds": n_seeds,
                        "match_ratio": match_ratio,
                        "method": "old",
                        "lambda": lamb,
                    }
                )

results = pd.DataFrame(rows)


#%%

from scipy.stats import wilcoxon


def compute_wilcoxon_pvalue(sub_results, ax):
    for i, n_seeds in enumerate(seeds_range):
        sub_sub_results = sub_results[sub_results["n_seeds"] == n_seeds]
        old_match_ratios = sub_sub_results[sub_sub_results["method"] == "old"][
            "match_ratio"
        ].values
        new_match_ratios = sub_sub_results[sub_sub_results["method"] == "new"][
            "match_ratio"
        ].values
        if (old_match_ratios - new_match_ratios).max() == 0:
            pvalue = 1
        else:
            _, pvalue = wilcoxon(old_match_ratios, new_match_ratios)
        if pvalue < 0.05:
            ax.text(i, 1, "*", transform=ax.transData, ha="center", va="bottom")


fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

for i, lamb in enumerate(lamb_range):
    ax = axs.flat[i]
    sub_results = results[results["lambda"] == lamb]
    sns.lineplot(
        data=sub_results,
        x="n_seeds",
        y="match_ratio",
        hue="method",
        ax=ax,
        legend=i == 0,
    )
    ax.set(title=f"lambda = {lamb:.1f}")
    # compute_wilcoxon_pvalue(sub_results, ax)
    # ax.set(ylim=(ax.get_ylim()[0], 1.05))

#%%
from scipy.sparse import csr_array


def generate_data(layered=False, sparse=False):
    A, B = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)

    perm = rng.permutation(n_side)
    undo_perm = np.argsort(perm)
    B = B[perm][:, perm]

    if sparse:
        A = csr_array(A)
        B = csr_array(B)

    if layered:
        A2, B2 = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
        B2 = B2[perm][:, perm]
        if sparse:
            A2 = csr_array(A2)
            B2 = csr_array(B2)
        return [A, A2], [B, B2], perm, undo_perm
    else:
        return A, B, perm, undo_perm


n_sims = 100
n_seeds = 1
lamb = 0.01

rows = []
for sparse in [False, True]:
    for layered in [False, True]:
        for _ in range(n_sims):
            A, B, perm, undo_perm = generate_data(layered=layered, sparse=sparse)

            if n_seeds > 0:
                seeds_A = np.random.choice(n_side, replace=False, size=n_seeds)
                # seeds_A = np.arange(n_seeds)
                seeds_B = np.argsort(perm)[seeds_A]
                partial_match = np.stack((seeds_A, seeds_B)).T
            else:
                seeds_A = seeds_B = []
                partial_match = None

            S = lamb * np.eye(n_side)
            S = np.random.uniform(0, 1, (n_side, n_side)) + S
            S = S[:, perm]

            solver = GraphMatchSolver(
                A,
                B,
                # similarity=S,
                partial_match=partial_match,
                use_numba=False,
            )
            solver.solve()
            match_ratio = (solver.permutation_ == undo_perm).mean()
            rows.append(
                {
                    "n_seeds": n_seeds,
                    "match_ratio": match_ratio,
                    "method": "new",
                    "lambda": lamb,
                    "sparse": sparse,
                    "layered": layered,
                }
            )
results = pd.DataFrame(rows)

#%%

results.groupby(["sparse", "layered"])["match_ratio"].mean()

#%% [markdown]
# ## Seeds out of order

#%%
contra_rho = 0.6
# simulate the correlated subgraphs
n_sims = 1000
rows = []
for n_seeds in [0, 1, 2, 3, 4, 5]:
    for sim in tqdm(range(n_sims), leave=False):
        if n_seeds > 0:
            seeds_A = np.random.choice(n_side, replace=False, size=n_seeds)
            seeds_B = np.argsort(perm)[seeds_A]
            partial_match = np.stack((seeds_A, seeds_B)).T
        else:
            partial_match = None
        A, B = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
        AB, BA = er_corr(n_side, contra_p, contra_rho, directed=True)
        solver = GraphMatchSolver(
            A, B, verbose=0, partial_match=partial_match, use_numba=True
        )
        solver.solve()
        match_ratio = (
            solver.permutation_[n_seeds:] == np.arange(len(A))[n_seeds:]
        ).mean()
        rows.append({"n_seeds": n_seeds, "match_ratio": match_ratio})
results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.lineplot(data=results, x="n_seeds", y="match_ratio", ax=ax)

gluefig("accuracy_by_seeds", fig)


#%% [markdown]
# ## Multilayer
# This simulation adds a second "layer" to each network. This can be thought of as an
# extra channel of information that might be helpful for matching - say, one layer could
# be a network of email communication, and another could be chat messages between the
# same actors.
#%%

# simulate the correlated subgraphs
n_sims = 1000
rows = []
for second_layer_rho in np.linspace(0, 1, 6):
    for sim in tqdm(range(n_sims), leave=False):
        A1, B1 = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
        A2, B2 = er_corr(n_side, contra_p, second_layer_rho, directed=True)
        solver = GraphMatchSolver([A1, A2], [B1, B2])
        solver.solve()
        match_ratio = (solver.permutation_ == np.arange(len(A))).mean()
        rows.append({"second_layer_rho": second_layer_rho, "match_ratio": match_ratio})
results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.lineplot(data=results, x="second_layer_rho", y="match_ratio", ax=ax)

gluefig("accuracy_by_seeds", fig)


#%% [markdown]
# ## Optimal transport
# Here, we employ the optimal transport approach of https://arxiv.org/abs/2111.05366 to
# solve an intermediate subproblem in the graph matching problem. The paper shows that
# this strategy can be both faster and more accurate, especially for larger networks.
#%%

# TODO: figure out the convergence warning stuff
import warnings

warnings.simplefilter("always")

# REF: Figure 2B of https://arxiv.org/abs/2111.05366
n_sims = 10
n = 250
p = np.log(n) / n
rows = []
for sim in tqdm(range(n_sims), leave=False):
    A, B = er_corr(n, p, 1.0)
    perm = rng.permutation(n)
    undo_perm = np.argsort(perm)
    B = B[perm][:, perm]
    for transport in [True, False]:
        # Note: transport_regularizer is higher than in the paper, was getting a conv.
        # warning
        solver = GraphMatchSolver(
            A,
            B,
            transport=transport,
            transport_regularizer=200,
            transport_implementation="pot",
            transport_tolerance=5e-2,
            transport_maxiter=1000,
        )
        solver.solve()
        match_ratio = (solver.permutation_ == undo_perm).mean()
        rows.append({"transport": transport, "match_ratio": match_ratio})
results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.barplot(data=results, x="transport", y="match_ratio", ax=ax)

#%% [markdown]
# ## Similarity
# This simulation adds some extra information in the form of a similarity matrix between
# true matches - in this case just an indicator function weighted by `lambda`.
#%%


n_sims = 1000
rows = []
for lamb in np.linspace(0, 1, 6):
    for sim in tqdm(range(n_sims), leave=False):

        A, B = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)

        perm = rng.permutation(n_side)
        undo_perm = np.argsort(perm)
        S = lamb * np.eye(B.shape[0])
        S = np.random.uniform(0, 1, (n_side, n_side)) + S
        S = S[:, perm]
        B = B[perm][:, perm]

        solver = GraphMatchSolver(A, B, S=S)
        solver.solve()
        match_ratio = (solver.permutation_ == undo_perm).mean()
        rows.append({"lambda": lamb, "match_ratio": match_ratio, "method": "new"})

        perm = GraphMatch().fit_predict(A, B, S=S)
        match_ratio = (perm == undo_perm).mean()
        match_ratio
        rows.append({"lambda": lamb, "match_ratio": match_ratio, "method": "old"})

results = pd.DataFrame(rows)

#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.lineplot(data=results, x="lambda", y="match_ratio", hue="method", ax=ax)

#%%

n_sims = 1000
rows = []
for lamb in np.linspace(0, 1, 6):
    for sim in tqdm(range(n_sims), leave=False):

        A, B = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
        S = lamb * np.eye(B.shape[0])
        perm = rng.permutation(n_side)
        undo_perm = np.argsort(perm)
        S = np.random.uniform(0, 1, (n_side, n_side)) + S
        S = S[:, perm]
        B = B[perm][:, perm]

        seeds = np.random.choice(n_side, replace=False, size=1)
        partial_match = np.stack((seeds, np.argsort(perm)[seeds])).T

        solver = GraphMatchSolver(A, B, S=S, partial_match=partial_match)
        solver.solve()
        match_ratio = (solver.permutation_ == undo_perm).mean()
        rows.append({"lambda": lamb, "match_ratio": match_ratio, "method": "new"})

        perm = GraphMatch().fit_predict(A, B, S=S, seeds_A=seeds, seeds_B=perm[seeds])
        match_ratio = (perm == undo_perm).mean()
        match_ratio
        rows.append({"lambda": lamb, "match_ratio": match_ratio, "method": "old"})

results = pd.DataFrame(rows)

#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.lineplot(data=results, x="lambda", y="match_ratio", hue="method", ax=ax)


#%% [markdown]
# ## End
#%%

elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")

#%%
A = [np.random.randint(1, size=(10, 10))]
B = [np.random.randint(1, size=(10, 10))]
C = np.random.randint(1, size=(10, 10))
D = np.random.randint(1, size=(10, 10))
P = np.random.uniform(1, size=(10, 10))

from numba import njit


def multiply(A, B, C, D, P):
    i = 0
    return A[i] @ P @ B[i].T


multiply(A, B, C, D, P)
