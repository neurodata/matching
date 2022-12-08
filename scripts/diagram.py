#%% [markdown]
# # Left/Right stereotypy in Kenyon cells
# Here we investigate whether the connections from projection neurons (PNs) to Kenyon
# cells (KCs) have a "stereotyped" structure. By stereotyped, we mean whether there is
# correlation in the edge structure between multiple samples of this subgraph. Here,
# we compare the left and right subgraphs for a single larval *Drosophila* connectome
# ([Eichler et al. (2017)](https://www.nature.com/articles/nature23455)).
#
# ## Introduction
# The general thinking in the field is that the connections from PNs to KCs are "random"
# (though there are are some caveats and debates, see
# [Zheng et al. (2020)](https://www.biorxiv.org/content/10.1101/2020.04.17.047167v2.abstract)).
# The word "random" is doing a lot of work here, and I don't think the field would agree
# to a distribution on that subgraph-as-a-random-variable that would satisfy us. The
# general idea is that the PNs are stereotyped and identifiable across animals: I can
# find the same one on each hemisphere of the brain, and across animals. Conversely, the
# KCs are not, because there is not thought to be correlation between the edges projecting
# from PNs to KCs. See the figure below for a schematic.

#%% [markdown]
# ```{figure} ./images/mittal-fig-2a.png
# ---
# width: 500px
# name: mb-schematic
# ---
# Schematic description of the mushroom body (here shown for the adult *Drosophila*,
# the larva has far fewer neurons). Connections from projection neurons (PNs) to
# Kenyon cells are thought to be random. Image from
# [Mittal et al. (2020)](https://www.nature.com/articles/s41467-020-14836-6).
# ```
#%%

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import histplot, matrixplot
from graspologic.plot import heatmap
from graspologic.simulations import er_corr
from graspologic.utils import binarize
from pkg.data import load_maggot_graph, load_unmatched, DATA_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

DISPLAY_FIGS = True

FILENAME = "diagram"


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)

left_adj, left_nodes = load_unmatched("left", weights=True)
right_adj, right_nodes = load_unmatched("right", weights=True)

mg = load_maggot_graph()

left_mg = mg.node_subgraph(mg[mg.nodes["left"]].nodes.index)
right_mg = mg.node_subgraph(mg[mg.nodes["right"]].nodes.index)

#%% [markdown]
# ## Data
# For this investigation, we select the subgraphs of connections from uniglomerular PNs
# to the multi-claw KCs, as these are the specific projections which are thought to be
# "unstructured" in their connectivity.
#
# We also remove any KCs which do not recieve a projection from one of these
# PNs.
#
# For this analysis, I will consider the subgraphs to be directed and unweighted (though
# I discuss the importance of weights going forward at the end).

#%%
nodes = left_mg.nodes
upns_left = nodes[nodes["merge_class"] == "uPN"].index
kcs_left = nodes[(nodes["class1"] == "KC") & (nodes["merge_class"] != "KC-1claw")].index

nodes = right_mg.nodes
upns_right = nodes[nodes["merge_class"] == "uPN"].index
kcs_right = nodes[
    (nodes["class1"] == "KC") & (nodes["merge_class"] != "KC-1claw")
].index


#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# left_color = "#66c2a5"
# right_color =
# left_color = "#66c2a5"
left_color = "#fc8d62"
pn_left_start = (0.1, 0.75)
kc_left_start = (0.5, 0.75)
kc_right_start = (0.7, 0.9)
pn_right_start = (0.9, 0.75)
n_pn = 10
n_kc = 10

from matplotlib.patches import Circle


def draw_dots(start, n, gap, color, hatch=None):
    positions = []
    for i in range(n):
        pos = start[0], start[1] - i * gap
        # ax.scatter(*pos, color=color)
        circ = Circle(pos, 0.02, facecolor=color, hatch=hatch, edgecolor="k", zorder=10)
        ax.add_patch(circ)
        positions.append(pos)
    return positions


left_pn_pos = draw_dots(pn_left_start, n_pn, 0.05, left_color)
left_kc_pos = draw_dots(kc_left_start, n_kc, 0.05, left_color, hatch="///")
# right_kc_pos = draw_dots(kc_right_start, n_kc, 0.1, right_color)
# right_pn_pos = draw_dots(pn_right_start, n_pn, 0.1, right_color)


def draw_edges(p, k, pos1s, pos2s, color):
    drawn_edges = set()
    for i, pos1 in enumerate(pos1s):
        edges = 0
        while edges < k:
            for j, pos2 in enumerate(pos2s):
                if rng.uniform() < p:
                    drawn_edges.add((i, j))
                    ax.plot(
                        (pos1[0], pos2[0]),
                        (pos1[1], pos2[1]),
                        color=color,
                        linewidth=1.5,
                    )
                    edges += 1
    return drawn_edges


def add_edges(drawn_edges, n, pos1s, pos2s, color):
    edge_count = 0
    while edge_count < n:
        i = rng.integers(n_pn)
        pos1 = pos1s[i]
        j = rng.integers(n_kc)
        pos2 = pos2s[j]
        if (i, j) not in drawn_edges:
            ax.plot(
                (pos1[0], pos2[0]),
                (pos1[1], pos2[1]),
                color=color,
                linewidth=3,
                linestyle="--",
            )
            edge_count += 1


rng = np.random.default_rng(888888)
drawn_edges = draw_edges(0.2, 3, left_pn_pos, left_kc_pos, left_color)
add_edges(drawn_edges, 3, left_pn_pos, left_kc_pos, "darkred")

# draw_edges(0.2, 7, right_pn_pos, right_kc_pos, right_color)

# perm = rng.permutation(n_kc)
# for i, target in enumerate(perm):
#     pos1 = left_kc_pos[i]
#     pos2 = right_kc_pos[target]
#     ax.plot(
#         (pos1[0], pos2[0]),
#         (pos1[1], pos2[1]),
#         color="grey",
#         linewidth=1,
#         zorder=-1,
#         linestyle="--",
#     )
fontsize = "large"
ax.text(pn_left_start[0], 0.2, "PNs", color=left_color, size=fontsize, ha="center")
ax.text(kc_left_start[0], 0.2, "LHNs", color=left_color, size=fontsize, ha="center")
# ax.text(pn_right_start[0], 0.25, "PNs", color=right_color, size=fontsize, ha="center")
# ax.text(kc_right_start[0], 0.1, "LHNs", color=right_color, size=fontsize, ha="center")
# ax.text(0.5, 0.5, r"$P$", ha="center", va="center", fontsize="xx-large")
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
gluefig("diagram", fig)

#%% [markdown]
# We also compute a metric to measure the degree of overlap between the matched
# subgraphs. This metric is called alignment strength, defined in
# [Fishkind et al. (2021)](https://link.springer.com/article/10.1007/s41109-021-00398-z).
# it measures the amount of edge disagreements relative to what one would expect by
# chance under a *random* matching.

#%%
observed_alignment = compute_alignment_strength_subgraph(A_sub, B_sub_perm)

glue("observed_alignment", observed_alignment)

#%% [markdown]
# We find that for the optimized matching, the alignment strength is
# {glue:text}`kc_stereotypy-observed_alignment:.2f`. But what should we make of this?
# Two random subgraphs would also have some degree of alignment between their edges
# under an optimized matching.

#%% [markdown]
# ## Comparing our edge disagreements to a null model
#
# To calibrate our expectations for the alignment strength, we compute the alignment
# strength for a series of network pairs sampled from a null model. In other words, we
# sample two networks *which share no edge correlation*, match them, and compute the
# alignment strength. This gives us a distribution of alignment strengths to compare
# to.

# %%

n_sims = 1000
glue("n_sims", n_sims)


def er_subgraph(size, p, rng=None):
    subgraph = rng.binomial(1, p, size=size)
    return subgraph


p_A = np.count_nonzero(A_sub) / A_sub.size
p_B = np.count_nonzero(B_sub) / B_sub.size

rows = []
for sim in tqdm(range(n_sims), leave=False):
    A_sim = er_subgraph(A_sub.shape, p_A, rng)
    B_sim = er_subgraph(B_sub.shape, p_B, rng)

    perm_inds, B_sim_perm = match_seeded_subgraphs(A_sim, B_sim)

    alignment = compute_alignment_strength_subgraph(A_sim, B_sim_perm)

    rows.append({"data": "ER", "alignment": alignment})

rows.append({"data": "Observed", "alignment": observed_alignment})

results = pd.DataFrame(rows)

#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
histplot(data=results, x="alignment", hue="data", kde=True, ax=ax)
ax.set(ylabel="", yticks=[], xlabel="Alignment strength")
ax.spines["left"].set_visible(False)
sns.move_legend(ax, loc="upper left")
gluefig("alignment_dist", fig)

#%% [markdown]
# ## Questions/thoughts
# ```{admonition} Note
# :class: note
# **I am using the "restricted-focus seeded graph matching" since I have a pair of bipartite networks where one "part" is
# completely seeded**
#
# In this case, the graph matching minimization problem reduces to to solving a linear
# assignment problem to do
#
# $$
# \min_P tr(P^T A_{12}^T B_{12})
# $$
#
# where $A_{12}$ is the subgraph of connections from seeded (PN) to nonseeded (KC) on
# one hemisphere, and $B_{12}$ is defined likewise for the other hemisphere. This is
# because $A_{11}, A_{21}$ and $A_{22}$ (and likewise for $B$) are all 0 due to how
# we've defined our subgraphs.
#
# ```
#
# ```{admonition} Question
# :class: tip
# **How exactly should I compute alignment strength here?**
# There are a couple of weirdnesses:
# - we have bipartite networks,
# - In the phantom alignment strength paper, it is suggested (I think) to only look at
#   the restricted alignment strength which considers the unseeded-to-unseeded subgraph.
#   But in our case, that subgraph is empty.
# ```
#
# ```{admonition} Question
# :class: tip
# **How to deal with weights for the null model?**
# ```
#
# ```{admonition} Question
# :class: tip
# **How to deal with weights for the alignment strength test statistic?**
# ```
#
# ```{admonition} Question
# :class: tip
# **How to deal with an unequal number of Kenyon cells?**
# ```

#%% [markdown]
# ## Weighted
#%%

from pkg.data import load_andre_subgraph

left_sub_df = load_andre_subgraph("left_mb_odorant")
print((left_sub_df.shape))

right_sub_df = load_andre_subgraph("right_mb_odorant")
print((right_sub_df.shape))

np.setxor1d(left_sub_df.index.values, right_sub_df.index.values)

right_sub_df = right_sub_df.iloc[:21]

right_sub_df = right_sub_df.reindex(left_sub_df.index)


#%%

A_sub = left_sub_df.values
B_sub = right_sub_df.values

A_pn_strength = np.sum(A_sub, axis=1)
B_pn_strength = np.sum(B_sub, axis=1)

A_node_data = pd.DataFrame(
    data=A_pn_strength, columns=["strength"], index=left_sub_df.index
).reset_index()
A_node_data["side"] = "Left"

B_node_data = pd.DataFrame(
    data=B_pn_strength, columns=["strength"], index=right_sub_df.index
).reset_index()
B_node_data["side"] = "Right"

node_data = pd.concat((A_node_data, B_node_data))
sort_index = node_data.groupby("PN")["strength"].mean().sort_values().index[::-1]

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.barplot(data=node_data, x="PN", y="strength", hue="side", ax=ax, order=sort_index)
sns.move_legend(
    ax, loc="upper right", bbox_to_anchor=(1, 1), frameon=True, title="Side"
)
ax.set_ylabel("Node strength")

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(
    data=node_data.pivot(index="PN", values="strength", columns="side"),
    x="Left",
    y="Right",
    ax=ax,
)

#%%
perm, B_sub_perm = match_seeded_subgraphs(A_sub, B_sub)
# B_sub_perm = B_sub[:, :A_sub.shape[1]]
ord = 2
stat_observed = np.linalg.norm(A_sub - B_sub_perm, ord=ord)

#%%


def plot_alignment(left, right, figsize=(8, 8)):
    diff = left - right
    fig, axs = plt.subplots(3, 1, figsize=figsize)

    matrixplot(left, cbar=False, ax=axs[0])
    matrixplot(right, cbar=False, ax=axs[1])
    matrixplot(diff, cbar=False, ax=axs[2])

    axs[0].set_ylabel("Left", rotation=0, ha="right")
    axs[1].set_ylabel("Right", rotation=0, ha="right")
    axs[2].set_ylabel("L - R", rotation=0, ha="right")

    fig.set_facecolor("w")

    return fig, axs


plot_alignment(A_sub, B_sub_perm)

gluefig("matched_subgraphs_weighted", fig)

#%%

weights_A = A_sub[np.nonzero(A_sub)]
weights_B = B_sub[np.nonzero(B_sub)]
p_A = compute_density_subgraph(A_sub)
p_B = compute_density_subgraph(B_sub)


def weighted_er_subgraph(size, p, weights, replace=True, rng=None):
    subgraph = er_subgraph(size, p, rng)
    row_inds, col_inds = np.nonzero(subgraph)
    n_edges = len(row_inds)
    sampled_weights = rng.choice(weights, size=n_edges, replace=replace)
    subgraph[row_inds, col_inds] = sampled_weights
    return subgraph


#%%

rows = []
for i in range(n_sims):
    A_sim = weighted_er_subgraph(A_sub.shape, p_A, weights_A, replace=True, rng=rng)
    B_sim = weighted_er_subgraph(B_sub.shape, p_B, weights_B, replace=True, rng=rng)
    perm, B_sim_perm = match_seeded_subgraphs(A_sim, B_sim)

    stat = np.linalg.norm(A_sim - B_sim_perm, ord=ord)
    rows.append({"data": "Random", "stat": stat})

rows.append({"data": "Observed", "stat": stat_observed})
results = pd.DataFrame(rows)

#%%

plot_alignment(A_sim, B_sim_perm)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
histplot(data=results, x="stat", hue="data", kde=True, ax=ax)
ax.set(ylabel="", yticks=[], xlabel="Difference norm")
ax.spines["left"].set_visible(False)

gluefig("alignment_dist_weighted", fig)

#%%
A_sub

from numba import njit


@njit
def swap_edge(adjacency, edge_list):
    orig_inds = np.random.choice(len(edge_list), size=2, replace=False)
    u, v = edge_list[orig_inds[0]]
    x, y = edge_list[orig_inds[1]]

    # edge already exists, hold
    if adjacency[u, y] == 1 or adjacency[x, v] == 1:
        return adjacency, edge_list

    # no possibility of self-loops in this case

    # don't need to check reverse edge, since directed

    # perform the swap
    adjacency[u, v] = 0
    adjacency[x, y] = 0

    adjacency[u, y] = 1
    adjacency[x, v] = 1

    # update edge list
    edge_list[orig_inds[0]] = [u, y]
    edge_list[orig_inds[1]] = [x, v]
    return adjacency, edge_list


def swap_edges(adjacency, n_swaps=1):
    adjacency = adjacency.copy()
    row_inds, col_inds = np.nonzero(adjacency)
    # TODO numba this for-loop?
    edge_list = np.array((row_inds, col_inds)).T
    for i in range(n_swaps):
        adjacency, edge_list = swap_edge(adjacency, edge_list)
    return adjacency


def apply_weights(adjacency, weights):
    row_inds, col_inds = np.nonzero(adjacency)
    weights = weights.copy()
    np.random.shuffle(weights)
    adjacency[row_inds, col_inds] = weights
    return adjacency


def generate_subgraph_samples(adjacency, n_samples=100, n_swaps=100000):
    adjacency = adjacency.copy()
    row_inds, col_inds = np.nonzero(adjacency)
    weights = adjacency[row_inds, col_inds].copy()
    adjacency[adjacency > 0] = 1

    samples = []
    for i in tqdm(range(n_samples)):
        adjacency = swap_edges(adjacency, n_swaps=n_swaps)
        adjacency = apply_weights(adjacency, weights)
        samples.append(adjacency.copy())

    return samples


from tqdm.autonotebook import tqdm
from random import shuffle

A_sub_samples = generate_subgraph_samples(A_sub, n_samples=100, n_swaps=1000000)
B_sub_samples = generate_subgraph_samples(B_sub, n_samples=100, n_swaps=1000000)
shuffle(B_sub_samples)
#%%
ord = 2
rows = []
for i in range(100):
    perm_inds, B_sub_perm = match_seeded_subgraphs(A_sub_samples[i], B_sub_samples[i])
    stat = np.linalg.norm(A_sub_samples[i] - B_sub_perm, ord=ord)
    rows.append({"data": "Random", "stat": stat})

perm, B_sub_perm = match_seeded_subgraphs(A_sub, B_sub)
stat_observed = np.linalg.norm(A_sub - B_sub_perm, ord=ord)
rows.append({"data": "Observed", "stat": stat_observed})
results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
histplot(data=results, x="stat", hue="data", kde=True, ax=ax)
ax.set(ylabel="", yticks=[], xlabel="Difference norm")
ax.spines["left"].set_visible(False)

gluefig("alignment_dist_weighted", fig)

#%% [markdown]
# ## Appendix
# ### Testing alignment strength implementation
#%%
# Seeded, bipartite alignment strength.

n = 20
p = 0.23
n_sims = 1_000
rows = []
for rho in [0.3, 0.8]:
    for permute in [True, False]:
        for i in tqdm(range(n_sims), leave=False):
            A1, B1 = er_corr(n, p, rho, loops=False, directed=True)
            A2, B2 = er_corr(n, p, rho, loops=False, directed=True)
            A3, B3 = er_corr(n, p, rho, loops=False, directed=True)
            A = np.hstack((A1, A2, A3))
            B = np.hstack((B1, B2, B3))
            if permute:
                permutation = np.random.permutation(B.shape[1])
                B = B[:, permutation]
            alignment = compute_alignment_strength_subgraph(A, B)
            rows.append({"permute": permute, "alignment": alignment, "rho": rho})

results = pd.DataFrame(rows)

#%%

fg = sns.FacetGrid(data=results, row="rho", col="permute", height=4, aspect=1.5)
fg.map(sns.kdeplot, "alignment", fill=True)


def meanline(x, *args, **kwargs):
    ax = plt.gca()
    ax.axvline(x.mean(), color="tab:blue")
    ax.spines["left"].set_visible(False)
    ax.set(ylabel="", yticks=[])


fg.map(meanline, "alignment")
fg.axes[0, 0].axvline(0.3, color="tab:orange")
fg.axes[0, 1].axvline(0.0, color="tab:orange")
fg.axes[1, 0].axvline(0.8, color="tab:orange")
_ = fg.axes[1, 1].axvline(0.0, color="tab:orange")

#%% [markdown]
# ## End
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
