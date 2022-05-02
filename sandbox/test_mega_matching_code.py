#%%
import graspologic as gl
import numpy as np
from scipy.optimize import linear_sum_assignment

xnp = np

n = 100
p = np.log(n) / n
rho = 0.6
maximize = True
if maximize:
    obj_func_scalar = -1
else:
    obj_func_scalar = 1

A_LL_1, A_RR_1 = gl.simulations.er_corr(n, p, rho, loops=False)
A_LR_1, A_RL_1 = gl.simulations.er_corr(n, p, rho, loops=False)
A_LL_2, A_RR_2 = gl.simulations.er_corr(n, p, rho, loops=False)
A_LR_2, A_RL_2 = gl.simulations.er_corr(n, p, rho, loops=False)

A_LL = [A_LL_1, A_LL_2]
A_RR = [A_RR_1, A_RR_2]
A_LR = [A_LR_1, A_LR_2]
A_RL = [A_RL_1, A_RL_2]

# A_LL = A_LL_1
# A_RR = A_RR_1
# A_LR = A_LR_1
# A_RL = A_RL_1

n_layers = len(A_LL)

from pkg.match import BisectedGraphMatchSolver

mean_match_ratio = 0
n_trials = 30
for i in range(n_trials):
    bgm = BisectedGraphMatchSolver(A_LL, A_RR, A_LR, A_RL)
    bgm.solve()
    match_ratio = (bgm.permutation_ == np.arange(n)).mean()
    mean_match_ratio += match_ratio / n_trials
mean_match_ratio

#%%
from pkg.match import GraphMatchSolver

gm = GraphMatchSolver(
    A_LL, A_RR, A_LR, A_RL, partial_match=np.array([[0, 0], [1, 1], [3, 3]])
)
