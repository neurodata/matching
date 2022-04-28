#%%
import graspologic as gl
import numpy as np
import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment


xnp = jnp

n = 100
p = np.log(n) / n
rho = 0.95
maximize = True
if maximize:
    obj_func_scalar = -1
else:
    obj_func_scalar = 1

A_LL_1, A_RR_1 = gl.simulations.er_corr(n, p, rho, loops=False)
A_LR_1, A_RL_1 = gl.simulations.er_corr(n, p, rho, loops=False)
A_LL_2, A_RR_2 = gl.simulations.er_corr(n, p, rho, loops=False)
A_LR_2, A_RL_2 = gl.simulations.er_corr(n, p, rho, loops=False)

A_LL_1 = xnp.array(A_LL_1)
A_RR_1 = xnp.array(A_RR_1)
A_LR_1 = xnp.array(A_LR_1)
A_RL_1 = xnp.array(A_RL_1)

A_LL_2 = xnp.array(A_LL_2)
A_RR_2 = xnp.array(A_RR_2)
A_LR_2 = xnp.array(A_LR_2)
A_RL_2 = xnp.array(A_RL_2)

A_LL = [A_LL_1, A_LL_2]
A_RR = [A_RR_1, A_RR_2]
A_LR = [A_LR_1, A_LR_2]
A_RL = [A_RL_1, A_RL_2]

n_layers = len(A_LL)


# P = xnp.full((n, n), 1 / n)
og_permutation = np.random.permutation(n)
P = 0.5 * np.eye(n)[og_permutation] + xnp.full((n, n), 1 / n)

n_iter = 0
while n_iter < 30:
    gradient = xnp.zeros((n, n))
    for i in range(n_layers):
        gradient += (
            A_LL[i] @ P @ A_RR[i].T
            + A_LL[i].T @ P @ A_RR[i]
            + A_LR[i] @ P.T @ A_RL[i].T
            + A_RL[i].T @ P.T @ A_LR[i]
        )

    _, permutation = linear_sum_assignment(gradient, maximize=maximize)
    Q = xnp.eye(n)[permutation]

    R = P - Q
    for i in range(n_layers):
        a_intra = xnp.trace(A_LL[i] @ R @ A_RR[i].T @ R.T)
        b_intra = xnp.trace(
            A_LL[i] @ Q @ A_RR[i].T @ R.T + A_LL[i] @ R @ A_RR[i].T @ Q.T
        )

        a_cross = xnp.trace(A_LR[i].T @ R @ A_RL[i] @ R)
        b_cross = xnp.trace(A_LR[i].T @ R @ A_RL[i] @ Q) + xnp.trace(
            A_LR[i].T @ Q @ A_RL[i] @ R
        )

    a = a_cross + a_intra
    b = b_cross + b_intra

    if a * obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
        alpha = -b / (2 * a)
    else:
        alpha = np.argmin([0, (b + a) * obj_func_scalar])

    P_new = alpha * P + (1 - alpha) * Q

    P = P_new
    n_iter += 1

_, permutation = linear_sum_assignment(P, maximize=maximize)

(permutation == np.arange(n)).mean()

# %%
