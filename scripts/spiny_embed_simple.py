#%%

import pandas as pd
from scipy.io import mmread


# thresholded by two or more connections


#%%
meta_loc = "matching/data/spiny/meta_adj_spiny_big.csv"
meta = pd.read_csv(meta_loc, index_col=0)
meta["bodyid"] = meta["bodyid"].astype(int)
meta = meta.set_index("bodyid")
#%%
adj_loc = "matching/data/spiny/adj_spiny_big.mtx"
adj_dense = mmread(adj_loc).toarray()
adj_df = pd.DataFrame(data=adj_dense, index=meta.index, columns=meta.index)
#%%
# 9A
# 6B
# 1A

#%%
select_meta = meta[meta["soma_neuromere"].isin(["T1"])]

select_adj = adj_df.loc[select_meta.index, select_meta.index]
# NBLAST on left to right mirrored data

# add the bilateral connections

#%%
# proofreading hemilineage

# 1. soft group constraints are of interest
# 2. how to do confidence of predictions

#%%

from graspologic.embed import LaplacianSpectralEmbed
from graspologic.utils import largest_connected_component, pass_to_ranks
from scipy.sparse import csr_matrix

#%%
adj_lcc, inds = largest_connected_component(select_adj.values, return_inds=True)
adj_lcc_ptr = pass_to_ranks(adj_lcc)
adj_to_embed = csr_matrix(adj_lcc_ptr)
#%%
lse = LaplacianSpectralEmbed(form="R-DAD", n_components=8, check_lcc=False)

X, Y = lse.fit_transform(adj_lcc_ptr)

#%%
meta_lcc = meta.iloc[inds]

#%%
import numpy as np

n_show = 3
embed_full = np.concatenate((X[:, :n_show], Y[:, :n_show]), axis=1)

#%%

labels = select_meta.iloc[inds]["soma_side"].values
labels = [x for x in labels]

from graspologic.plot import pairplot

pairplot(embed_full, alpha=0.2, labels=labels)

#%%

labels = list(select_meta.iloc[inds]["class"])
pairplot(embed_full, alpha=0.2, labels=labels)

#%%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP

from graspologic.embed import select_dimension

# from .utils import legend_upper_right, soft_axis_off


def simple_scatterplot(
    X,
    labels=None,
    palette="deep",
    ax=None,
    title="",
    legend=False,
    figsize=(10, 10),
    s=15,
    alpha=0.7,
    linewidth=0,
    spines_off=True,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    plot_df = pd.DataFrame(data=X[:, :2], columns=["0", "1"])
    plot_df["labels"] = labels
    sns.scatterplot(
        data=plot_df,
        x="0",
        y="1",
        hue="labels",
        palette=palette,
        ax=ax,
        s=s,
        alpha=alpha,
        linewidth=linewidth,
        **kwargs,
    )
    ax.set(title=title)
    # if spines_off:
    #     soft_axis_off(ax, top=False, bottom=False, right=False, left=False)
    # else:
    #     soft_axis_off(ax, top=False, bottom=True, right=False, left=True)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    if legend:
        # convenient default that I often use, places in the top right outside of plot
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    return ax


def simple_umap_scatterplot(
    X,
    labels=None,
    min_dist=0.75,
    n_neighbors=20,
    metric="euclidean",
    umap_kws={},
    palette="deep",
    ax=None,
    title="",
    legend=False,
    figsize=(10, 10),
    s=15,
    alpha=0.7,
    linewidth=0,
    scatter_kws={},
):
    umapper = UMAP(
        min_dist=min_dist, n_neighbors=n_neighbors, metric=metric, **umap_kws
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="umap")
    umap_embedding = umapper.fit_transform(X)
    ax = simple_scatterplot(
        umap_embedding,
        labels=labels,
        palette=palette,
        ax=ax,
        title=r"UMAP $\circ$ " + title,
        legend=legend,
        figsize=figsize,
        s=s,
        alpha=alpha,
        linewidth=linewidth,
        **scatter_kws,
    )
    return ax


#%%
from umap import UMAP

# from giskard import simple_scatterplot

# umapper = UMAP()

simple_umap_scatterplot(embed_full, labels=labels)

#%%


embed_full = np.concatenate((X, Y), axis=1)
min_dist = 0.75
n_neighbors = 20
metric = "cosine"
umapper = UMAP(min_dist=min_dist, n_neighbors=n_neighbors, metric=metric)
# warnings.filterwarnings("ignore", category=UserWarning, module="umap")
umap_embedding = umapper.fit_transform(embed_full)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
labels = list(select_meta.iloc[inds]["soma_side"])
simple_scatterplot(umap_embedding, labels=labels, ax=ax, legend=True)

# try doing the left right collapse
#%%


def select_paired_neuromere(meta, neuromere):
    # select the neuromere
    select_meta = meta[meta["soma_neuromere"].isin([neuromere])]

    # count occurences of "group"
    group_counts = select_meta["group"].value_counts()

    # find everything that occurs only twice
    single_group_counts = group_counts[group_counts == 2].index

    # subselect to these which are paired
    select_meta = select_meta[select_meta["group"].isin(single_group_counts)]
    select_meta = select_meta.sort_values("group")

    # split left right
    left_meta = select_meta[select_meta["soma_side"] == "LHS"]
    right_meta = select_meta[select_meta["soma_side"] == "RHS"]

    # grab subset for which we have on both left and right
    # because of filter above, these are the cases where we have one on right and left
    groups_in_both = np.intersect1d(left_meta["group"], right_meta["group"])
    left_meta = left_meta[left_meta["group"].isin(groups_in_both)]
    right_meta = right_meta[right_meta["group"].isin(groups_in_both)]

    # ensure indexed the same, useful assumption later
    assert (left_meta["group"].values == right_meta["group"].values).all()

    return left_meta, right_meta


left_paired_meta, right_paired_meta = select_paired_neuromere(meta_lcc, "T1")

select_meta = meta[meta["soma_neuromere"].isin(["T1"])]
select_adj = adj_df.loc[select_meta.index, select_meta.index]
adj_lcc, lcc_inds = largest_connected_component(select_adj.values, return_inds=True)
meta_lcc = meta.iloc[lcc_inds]

left_meta = meta_lcc[meta_lcc["soma_side"] == "LHS"]
right_meta = meta_lcc[meta_lcc["soma_side"] == "RHS"]

adj_lcc_ptr = pass_to_ranks(adj_lcc)
adj_to_embed = csr_matrix(adj_lcc_ptr)

index_map = dict(zip(meta_lcc.index, np.arange(len(meta_lcc))))
left_inds = left_meta.index.map(index_map)
right_inds = right_meta.index.map(index_map)

left_index_map = dict(zip(left_meta.index, np.arange(len(left_meta))))
left_paired_inds = left_paired_meta.index.map(left_index_map)

right_index_map = dict(zip(right_meta.index, np.arange(len(right_meta))))
right_paired_inds = right_paired_meta.index.map(right_index_map)

#%%

lse = LaplacianSpectralEmbed(form="R-DAD", n_components=8, check_lcc=False)
X, Y = lse.fit_transform(adj_lcc_ptr)

X_left = X[left_inds]
X_right = X[right_inds]
Y_left = Y[left_inds]
Y_right = Y[right_inds]

from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
import time

#%%
def joint_procrustes(
    data1,
    data2,
    method="orthogonal",
    seeds=None,
    swap=False,
    verbose=False,
):
    n = len(data1[0])
    if method == "orthogonal":
        procruster = OrthogonalProcrustes()
    elif method == "transport":
        if seeds is None:
            procruster = SeedlessProcrustes(init="sign_flips")
        else:
            paired_inds1 = seeds[0]
            paired_inds2 = seeds[1]
            X1_paired = data1[0][paired_inds1, :]
            X2_paired = data2[0][paired_inds2, :]
            if swap:
                Y1_paired = data1[1][paired_inds2, :]
                Y2_paired = data2[1][paired_inds1, :]
            else:
                Y1_paired = data1[1][paired_inds1, :]
                Y2_paired = data2[1][paired_inds2, :]
            data1_paired = np.concatenate((X1_paired, Y1_paired), axis=0)
            data2_paired = np.concatenate((X2_paired, Y2_paired), axis=0)
            op = OrthogonalProcrustes()
            op.fit(data1_paired, data2_paired)
            procruster = SeedlessProcrustes(
                init="custom",
                initial_Q=op.Q_,
                optimal_transport_eps=1.0,
                optimal_transport_num_reps=100,
                iterative_num_reps=10,
            )
    data1 = np.concatenate(data1, axis=0)
    data2 = np.concatenate(data2, axis=0)
    currtime = time.time()
    data1_mapped = procruster.fit_transform(data1, data2)
    if verbose > 1:
        print(f"{time.time() - currtime:.3f} seconds elapsed for SeedlessProcrustes.")
    data1 = (data1_mapped[:n], data1_mapped[n:])
    return data1


X_left_mapped, Y_left_mapped = joint_procrustes(
    (X_left, Y_left),
    (X_right, Y_right),
    method="transport",
    seeds=(left_paired_inds, right_paired_inds),
)

#%%
X_mapped = np.concatenate((X_left_mapped, X_right), axis=0)
Y_mapped = np.concatenate((Y_left_mapped, Y_right), axis=0)
Z = np.concatenate((X_mapped, Y_mapped), axis=1)
stacked_meta = pd.concat((left_meta, right_meta))
labels = stacked_meta["soma_side"].values
pairplot(Z[:, :3], labels=labels)

#%%

Xpl = X[left_paired_inds]
Xpr = X[right_paired_inds]
labels = len(Xpl) * ["L"] + len(Xpr) * ["R"]
Xtest = np.concatenate((Xpl, Xpr), axis=0)
pairplot(Xtest[:, :3], labels=labels)
from graspologic.align import OrthogonalProcrustes

op = OrthogonalProcrustes()
Xpl_mapped = op.fit_transform(Xpl, Xpr)
Xtest = np.concatenate((Xpl_mapped, Xpr), axis=0)
pairplot(Xtest[:, :3], labels=labels)
