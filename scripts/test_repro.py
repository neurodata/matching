# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

#%%

import random
import unittest

import numpy as np

from graspologic.match import graph_match
from graspologic.simulations import er_np, sbm_corr


# np.random.seed(888)
# n = 10
# p = 0.2
# A = er_np(n=n, p=p)
# B = A.copy()
# permutation = np.random.permutation(n)
# B = B[permutation][:, permutation]
# _, indices_B, _, _ = graph_match(A, B, rng=999)
# for i in range(10):
#     # this fails w/o rng set here; i.e. there is variance
#     _, indices_B_repeat, _, _ = graph_match(A, B)
#     print(np.array_equal(indices_B, indices_B_repeat))

# %%
A = [
    [0, 90, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [90, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0],
    [10, 0, 0, 0, 43, 0, 0, 0, 0, 0, 0, 0],
    [0, 23, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0],
    [0, 0, 43, 0, 0, 0, 26, 0, 0, 0, 0, 0],
    [0, 0, 0, 88, 0, 0, 0, 16, 0, 0, 0, 0],
    [0, 0, 0, 0, 26, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 16, 0, 0, 0, 96, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 29, 0],
    [0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 37],
    [0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0],
]
B = [
    [0, 36, 54, 26, 59, 72, 9, 34, 79, 17, 46, 95],
    [36, 0, 73, 35, 90, 58, 30, 78, 35, 44, 79, 36],
    [54, 73, 0, 21, 10, 97, 58, 66, 69, 61, 54, 63],
    [26, 35, 21, 0, 93, 12, 46, 40, 37, 48, 68, 85],
    [59, 90, 10, 93, 0, 64, 5, 29, 76, 16, 5, 76],
    [72, 58, 97, 12, 64, 0, 96, 55, 38, 54, 0, 34],
    [9, 30, 58, 46, 5, 96, 0, 83, 35, 11, 56, 37],
    [34, 78, 66, 40, 29, 55, 83, 0, 44, 12, 15, 80],
    [79, 35, 69, 37, 76, 38, 35, 44, 0, 64, 39, 33],
    [17, 44, 61, 48, 16, 54, 11, 12, 64, 0, 70, 86],
    [46, 79, 54, 68, 5, 0, 56, 15, 39, 70, 0, 18],
    [95, 36, 63, 85, 76, 34, 37, 80, 33, 86, 18, 0],
]
A, B = np.array(A), np.array(B)

n = len(A)
pi = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - [1] * n
custom_init = np.eye(n)
custom_init = custom_init[pi]

_, indices_B, score, _ = graph_match(
    A, B, maximize=False, init=custom_init, shuffle_input=False
)
print(indices_B)
print(pi)
