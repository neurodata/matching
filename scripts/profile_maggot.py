#%%

import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import upset_catplot
from git import Repo
from graspologic.match import graph_match
from pkg.data import load_matched
from pkg.io import FIG_PATH, OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from tqdm.autonotebook import tqdm

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

#%%
left_adj, left_meta = load_matched(side="left", weights=True)
right_adj, right_meta = load_matched(side="right", weights=True)
left_nodes = left_meta.index
right_nodes = right_meta.index

data_dir = Path("matching/data/2021-05-24-v2/")
graph_types = ["Gaa", "Gad", "Gda", "Gdd"]
data = (("weight", float),)
gs_by_type = {}
adjs_by_type = {}
for g_type in graph_types:
    g = nx.read_edgelist(
        data_dir / f"{g_type}_edgelist.txt",
        create_using=nx.DiGraph,
        delimiter=" ",
        nodetype=int,
        data=data,
    )
    gs_by_type[g_type] = g
    adj_df = nx.to_pandas_adjacency(g)
    adjs_by_type[g_type] = adj_df

lls = []
rrs = []
lrs = []
rls = []
for g_type in graph_types:
    adj_df = adjs_by_type[g_type]
    lls.append(adj_df.reindex(index=left_nodes, columns=left_nodes).fillna(0.0).values)
    rrs.append(
        adj_df.reindex(index=right_nodes, columns=right_nodes).fillna(0.0).values
    )
    lrs.append(adj_df.reindex(index=left_nodes, columns=right_nodes).fillna(0.0).values)
    rls.append(adj_df.reindex(index=right_nodes, columns=left_nodes).fillna(0.0).values)
n = len(left_nodes)

#%%

n_seeds = 0
n_trials = 10

RERUN = False

if RERUN:

    rows = []
    with tqdm(total=2 * 2 * n_trials) as pbar:
        for bilateral in [False, True]:
            for multilayer in [False, True]:
                if multilayer:
                    A = lls
                    B = rrs
                    AB = lrs
                    BA = rls
                else:
                    A = np.sum(lls, axis=0)
                    B = np.sum(rrs, axis=0)
                    AB = np.sum(lrs, axis=0)
                    BA = np.sum(rls, axis=0)

                if not bilateral:
                    AB = None
                    BA = None

                for trial in range(n_trials):
                    currtime = time.time()
                    indices_l, indices_r, score, misc = graph_match(
                        A, B, AB=AB, BA=BA, rng=rng
                    )
                    elapsed = time.time() - currtime
                    match_ratio = (indices_l == indices_r).mean()

                    result = {
                        "match_ratio_full": match_ratio,
                        "n_seeds": n_seeds,
                        "elapsed": elapsed,
                        "bilateral": bilateral,
                        "n_iter": misc[0]["n_iter"],
                        "converged": misc[0]["converged"],
                        "n": n,
                        "score": score,
                        "multilayer": multilayer,
                    }
                    rows.append(result)
                    print(
                        f"bilateral={bilateral}, multilayer={multilayer}, match_ratio={match_ratio:.2f}"
                    )
                    pbar.update()
    results = pd.DataFrame(rows)
    results.to_csv(OUT_PATH / "matching_summary.csv")
else:
    results = pd.read_csv(OUT_PATH / "matching_summary.csv", index_col=0)

results


# %%


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
y_var = "match_ratio_full"
sns.stripplot(
    data=results,
    x="bilateral",
    hue="multilayer",
    dodge=True,
    y=y_var,
    ax=ax,
)
ax.set_ylabel("")
ax.set_xlabel("Bilateral connections")

for i, bilateral in enumerate([False, True]):
    for j, multilayer in enumerate([False, True]):
        sub_results = results[
            (results["multilayer"] == multilayer) & (results["bilateral"] == bilateral)
        ]
        y = sub_results["match_ratio_full"].mean()
        x_low = i + j * 0.4 - 0.3
        x_high = i + j * 0.4 - 0.1
        ax.plot([x_low, x_high], [y, y], color="black")
        ax.text(x_high + 0.02, y, f"{y:.2f}", va="center")
ax.set(ylabel="Matching accuracy")
sns.move_legend(ax, loc="lower right", frameon=True, title="Multilayer")
gluefig("matching_accuracy", fig)


#%%


colors = sns.color_palette()
ucplot = upset_catplot(
    results,
    x=["bilateral", "multilayer"],
    y="match_ratio_full",
    kind="strip",
    s=7,
    jitter=0.18,
    upset_size=50,
    upset_linewidth=3,
    color=colors[0],
    estimator=np.mean,
    estimator_labels=True,
    figsize=(8, 7),
)
ucplot.set_upset_ticklabels(["With edge types", "With contralateral"])
ucplot.set_ylabel("Matching accuracy")
gluefig("matching_accuracy_upset", ucplot.fig)

#%%

rerun = False
bilateral = True
multilayer = True
A = lls
B = rrs
AB = lrs
BA = rls
n_trials = 5
n_seeds_range = [50, 100, 200, 500]
if rerun:
    rows = []
    with tqdm(total=len(n_seeds_range) * n_trials) as pbar:
        for n_seeds in n_seeds_range:
            for trial in range(n_trials):
                seeds = rng.choice(n, size=n_seeds, replace=False)
                partial_match = np.column_stack((seeds, seeds))
                nonseeds = np.setdiff1d(np.arange(n), seeds)

                currtime = time.time()
                indices_l, indices_r, score, misc = graph_match(
                    A, B, AB=AB, BA=BA, partial_match=partial_match
                )
                elapsed = time.time() - currtime
                match_ratio_full = (np.arange(n) == indices_r).mean()

                match_ratio_restricted = (nonseeds == indices_r[nonseeds]).mean()

                result = {
                    "match_ratio_full": match_ratio_full,
                    "match_ratio_restricted": match_ratio_restricted,
                    "n_seeds": n_seeds,
                    "elapsed": elapsed,
                    "bilateral": bilateral,
                    "n_iter": misc[0]["n_iter"],
                    "converged": misc[0]["converged"],
                    "n": n,
                    "score": score,
                    "multilayer": multilayer,
                }
                rows.append(result)
                print(
                    f"bilateral={bilateral}, multilayer={multilayer}, match_ratio_full={match_ratio_full:.2f}"
                )
                pbar.update()
        results = pd.DataFrame(rows)
        results.to_csv(OUT_PATH / "matching_seeded.csv")
else:
    results = pd.read_csv(OUT_PATH / "matching_seeded.csv", index_col=0)

results

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=results, x="n_seeds", y="match_ratio_restricted", ax=ax)
sns.scatterplot(data=results, x="n_seeds", y="match_ratio_restricted", ax=ax)
ax.set(ylabel="Matching accuracy", xlabel="Number of seeds")
gluefig("matching_seeds", fig)
