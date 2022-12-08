#%%
from scipy.io import mmread
import pandas as pd
from graspologic.match import graph_match
from pathlib import Path
import numpy as np

data_path = Path("matching/data/spiny/email-2022-09-07")
adj_t1 = mmread(data_path / "mancT1.mtx").toarray()
adj_t2 = mmread(data_path / "mancT2.mtx").toarray()
adj_t1 = (adj_t1 > 0) * 1
adj_t2 = (adj_t2 > 0) * 1
lab_t1 = pd.read_csv(data_path / "mancT1.csv")
lab_t2 = pd.read_csv(data_path / "mancT2.csv")


def makes_seeds_ready(labels_t1, labels_t2, neuromere1, neuromere2):
    seeds_left = seeds[seeds.soma_neuromere == neuromere1]
    seeds_right = seeds[seeds.soma_neuromere == neuromere2]
    seeds1, seeds2 = [], []
    seed_groups = []
    n_small = min((seeds_left.shape[0], seeds_right.shape[0]))
    for i in range(n_small):
        if np.any(labels_t1.serial == seeds_left.iloc[i].serial) and np.any(
            labels_t2.serial == seeds_left.iloc[i].serial
        ):
            if seeds_left.iloc[i].serial in seed_groups:
                continue
            seeds1.append(np.where(labels_t1.serial == seeds_left.iloc[i].serial)[0][0])
            seeds2.append(np.where(labels_t2.serial == seeds_left.iloc[i].serial)[0][0])
            seed_groups.append(seeds_left.iloc[i].serial)
    return seeds1, seeds2, seed_groups


seeds1, seeds2, seed_groups = makes_seeds_ready(lab_t1, lab_t2, "T1", "T2")
#%%
gm = GraphMatch(
    n_init=1,
    init="barycenter",
    max_iter=30,
    shuffle_input=True,
    eps=1e-3,
    gmp=True,
    padding="adopted",
)

t0 = time.time()
gm.fit(adj_t1, adj_t2, seeds1, seeds2)
print(f"Elapsed time: {time.time()-t0} sec")
perm_inds = gm.perm_inds_
print(f"Matching objective function: {gm.score_}")
