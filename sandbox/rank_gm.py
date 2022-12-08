#%%
from graspologic.utils import remove_loops
import numpy as np
from graspologic.match import graph_match
from pkg.data import load_maggot_graph


def signal_flow(A):
    """Implementation of the signal flow metric from Varshney et al 2011

    Parameters
    ----------
    A : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    A = A.copy()
    A = remove_loops(A)
    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    b = np.sum(W * np.sign(A - A.T), axis=1)
    L_pinv = np.linalg.pinv(L)
    z = L_pinv @ b

    return z


def rank_signal_flow(A):
    sf = signal_flow(A)
    perm_inds = np.argsort(-sf)
    return perm_inds


def rank_graph_match_flow(A, n_init=10, max_iter=30, **kwargs):
    n = len(A)
    try:
        initial_perm = rank_signal_flow(A)
        init = np.eye(n)[initial_perm]
    except np.linalg.LinAlgError:
        print("SVD did not converge in signal flow")
        init = np.full((n, n), 1 / n)
    match_mat = np.zeros((n, n))
    triu_inds = np.triu_indices(n, k=1)
    match_mat[triu_inds] = 1
    _, perm_inds, _, _ = graph_match(
        match_mat, A, n_init=n_init, max_iter=max_iter, init=init, **kwargs
    )
    return perm_inds


#%%
mg = load_maggot_graph()

# %%
nodes = mg.nodes.copy()
nodes = nodes[~nodes["sum_walk_sort"].isna()]
nodes.sort_values("sum_walk_sort", inplace=True)
nodes["order"] = range(len(nodes))
mg = mg.node_subgraph(nodes.index)
mg.nodes = nodes

#%%
sum_adj = mg.sum.adj

#%%
perm_inds = rank_graph_match_flow(sum_adj, n_init=1, n_jobs=1, verbose=True)

# %%
perm_adjs = [adj[perm_inds][:, perm_inds] for adj in mg.adjs]

#%%

import seaborn as sns
import matplotlib.pyplot as plt
from pkg.plot import set_theme

set_theme()

colors = sns.color_palette("Paired", 10)

palette = {
    "aa-upper": colors[7],
    "aa-lower": colors[6],
    "ad-upper": colors[1],
    "ad-lower": colors[0],
    "dd-upper": colors[3],
    "dd-lower": colors[2],
    "da-upper": colors[5],
    "da-lower": colors[4],
    "sum-upper": colors[9],
    "sum-lower": colors[8],
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
print("Edge type: feedforward proportion")

order_map = {"sum": 0, "ad": 1, "aa": 2, "dd": 3, "da": 4}

for i, (adj, name) in enumerate(zip(perm_adjs, mg.edge_types[:])):
    upper_mass = adj[np.triu_indices_from(adj, k=1)].mean()
    lower_mass = adj[np.tril_indices_from(adj, k=1)].mean()
    upper_mass_prop = upper_mass / (upper_mass + lower_mass)
    lower_mass_prop = lower_mass / (upper_mass + lower_mass)
    print(f"{name}: {upper_mass_prop:.2f}")
    x = order_map[name]
    ax.bar(x - 0.2, upper_mass_prop, width=0.4, color=palette[f"{name}-upper"])
    ax.bar(x + 0.2, lower_mass_prop, width=0.4, color=palette[f"{name}-lower"])


ax.set_xticks(range(5))
ax.set_xticklabels(
    [
        "Sum",
        r"A $\rightarrow$ D",
        r"A $\rightarrow$ A",
        r"D $\rightarrow$ D",
        r"D $\rightarrow$ A",
    ],
    fontsize=15,
)
ax.set_ylabel("Proportion of synapses")
ax.set_ylim((0, 1))
plt.savefig("ffwd-fdback", dpi=300)
plt.savefig("ffwd-fdback.svg")

#%%
sorted_nodes = nodes.iloc[perm_inds].copy()

# sorted_nodes = nodes.copy().sort_values("sum_signal_flow", ascending=False)

sorted_nodes["rank"] = np.arange(len(sorted_nodes)) / len(sorted_nodes)

fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
name_map = {"input": "Input", "output": "Output", "inter": "Inter"}

for i, (name, group) in enumerate(sorted_nodes.groupby("io")):
    ax = axs[i]
    sns.histplot(group["rank"], ax=axs[i], stat="frequency")
    ax.spines["left"].set_visible(False)
    ax.set(yticks=[], ylabel=name_map[name])

axs[2].set_xlabel("Rank")
plt.savefig('rank-hist', dpi=300)
plt.savefig('rank-hist.svg')