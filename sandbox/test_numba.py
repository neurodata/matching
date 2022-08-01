#%%

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.match import graph_match  # experimental version
from graspologic.match import GraphMatch
from graspologic.simulations import er_corr
from pkg.io import FIG_PATH, OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from tqdm.autonotebook import tqdm


A, B = er_corr(10, 0.2, 0.9, directed=True, loops=False)
A = A.astype(float)
B = B.astype(float)
graph_match(A, B, use_numba=True)
