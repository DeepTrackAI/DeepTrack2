#%%
import deeptrack as dt
 
source = dt.sources.Source(a=[1, 2], b=[3, 4])
source = source.product(c=[5, 6])
source._dict
