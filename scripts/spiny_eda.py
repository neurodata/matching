#%%
import time

import numpy as np
import pandas as pd
from graspologic.partition import leiden
from graspologic.utils import largest_connected_component, symmetrize
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

# %%
adj = adj_df.values
adj_lcc, inds = largest_connected_component(adj, return_inds=True)
adj_lcc_undirected = symmetrize(adj_lcc)
meta_lcc = meta.iloc[inds].copy()

#%%
currtime = time.time()
partition_map = leiden(adj_lcc_undirected)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
pos_index = np.arange(len(meta_lcc))
comm_labels = np.vectorize(partition_map.get)(pos_index)
meta_lcc["leiden_labels"] = comm_labels

#%%
from giskard.plot import crosstabplot
import seaborn as sns
import colorcet as cc


def plot_attribute(name, meta):
    meta = meta.copy()
    series = meta[name].fillna("unk")
    meta[name] = series
    vals = np.unique(series)
    if len(vals) > 20:
        colors = cc.glasbey_light
    else:
        colors = sns.color_palette("tab20")
    palette = dict(zip(vals, colors))
    crosstabplot(meta, group="leiden_labels", hue=name, palette=palette)


plot_attribute("soma_neuromere", meta_lcc)
plot_attribute("soma_side", meta_lcc)
plot_attribute("hemilineage", meta_lcc)
