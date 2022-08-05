#%%

from pkg.data import load_split_connectome
import numpy as np

#%%
adj, nodes = load_split_connectome("herm_chem")

#%%
nodes = nodes.copy()
nodes["_inds"] = np.arange(len(nodes))
left_nodes = nodes[nodes["hemisphere"] == "L"]
right_nodes = nodes[nodes["hemisphere"] == "R"]
assert (left_nodes["pair"].values == right_nodes["pair"].values).all()
left_inds = left_nodes["_inds"].values
right_inds = right_nodes["_inds"].values

#%%
A_ll = adj[left_inds][:, left_inds]
A_rr = adj[right_inds][:, right_inds]
A_lr = adj[left_inds][:, right_inds]
A_rl = adj[right_inds][:, left_inds]

# %%
from graspologic.plot import heatmap
import matplotlib.pyplot as plt
from pkg.plot import set_theme

set_theme()


def multi_heatmap(A_ll, A_lr, A_rl, A_rr, title=""):
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(10, 10),
        gridspec_kw=dict(wspace=0, hspace=0.05),
        sharex=True,
        sharey=True,
    )

    heatmap_kws = dict(cbar=False, transform="simple-nonzero")
    heatmap(A_ll, ax=axs[0, 0], **heatmap_kws)
    heatmap(A_lr, ax=axs[0, 1], **heatmap_kws)
    heatmap(A_rl, ax=axs[1, 0], **heatmap_kws)
    heatmap(A_rr, ax=axs[1, 1], **heatmap_kws)

    axs[0, 0].set_ylabel("Left")
    axs[0, 0].set_title("Left")
    axs[0, 1].set_title("Right")
    axs[1, 0].set_ylabel("Right")

    fig.set_facecolor("w")
    fig.text(0.51, 0.92, title, ha="center", va="bottom", fontsize="x-large")

    return fig, axs


multi_heatmap(A_ll, A_lr, A_rl, A_rr, title='Known permutation')

# %%

rng = np.random.default_rng(888)

random_permutation = rng.permutation(A_rr.shape[0])

random_permutation

#%%


def apply_permutation(permutation, A_lr, A_rl, A_rr):
    A_lr = A_lr[:, permutation]
    A_rl = A_rl[permutation]
    A_rr = A_rr[permutation][:, permutation]
    return A_lr, A_rl, A_rr


A_lr_rand, A_rl_rand, A_rr_rand = apply_permutation(
    random_permutation, A_lr, A_rl, A_rr
)

multi_heatmap(A_ll, A_lr_rand, A_rl_rand, A_rr_rand, title="Random permutation")


#%%

from graspologic.match import graph_match  # note: this is the new experimental code/API
import time

currtime = time.time()
indices_A, indices_B, score, misc = graph_match(
    A_ll,
    A_rr_rand,
    AB=A_lr_rand,
    BA=A_rl_rand,
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

#%%
indices_A

#%%
indices_B

#%%
score

#%%
misc

#%%
currtime = time.time()
indices_A, indices_B, score, misc = graph_match(
    A_ll, A_rr_rand, AB=A_lr_rand, BA=A_rl_rand, max_iter=200
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")
print(f"Converged: {misc[0]['converged']}")

#%%
score

#%%

A_lr_fit, A_rl_fit, A_rr_fit = apply_permutation(
    indices_B, A_lr_rand, A_rl_rand, A_rr_rand
)

multi_heatmap(A_ll, A_lr_fit, A_rl_fit, A_rr_fit, title="Fit permutation")

#%%
undo_permutation = np.argsort(random_permutation)
match_ratio = (indices_B == undo_permutation).mean()
match_ratio
