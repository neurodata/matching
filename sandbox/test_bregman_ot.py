#%%
import ot

a = [0.5, 0.5]
b = [0.5, 0.5]
M = [[0.0, 1.0], [1.0, 0.0]]
ot.sinkhorn(a, b, M, 1)

# %%
