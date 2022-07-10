#%%
import deeptrack as dt
import numpy as np
import gc
import sys

v = dt.Value(lambda: np.zeros((2, 1)))

g = dt.generators.ContinuousGenerator(v, min_data_size=10, max_data_size=11)

with g:
    ...

#%%
r = gc.get_referrers(g.data)
len(r)
# %%
