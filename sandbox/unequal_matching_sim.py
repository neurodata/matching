#%%

from graspologic.match import graph_match
import numpy as np

A = np.random.rand(10, 10)
B = np.random.rand(11, 11)
S = np.eye(10, 11) * 10

_, perm_B, _, misc = graph_match(A, B, S=S)
score = misc[0]["score"]

(perm_B == np.arange(10)).mean()

out_score = np.sum(A * B[perm_B][:, perm_B]) + np.trace(S[:, perm_B])

#%%
