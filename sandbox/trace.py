#%%
import numpy as np
from scipy.sparse import csr_array

A = np.random.uniform(size=(4, 4))
permutation = np.random.permutation(A.shape[0])
print(np.trace(A[:, permutation]))
B = csr_array(A)
print(B[:, permutation].trace())
print(np.sum(B[:, permutation]))
print(np.sum(S[:, permutation].diagonal()))