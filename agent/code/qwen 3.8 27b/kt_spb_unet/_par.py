import time, warnings; warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import sys; sys.path.insert(0,"/a/kt_spb_unet")
import config, generator

def _init():
    generator._make_green_optics(config.HEIGHT, config.WIDTH)
    generator._make_red_optics(config.HEIGHT, config.WIDTH)

def _work(i):
    h, w = config.HEIGHT, config.WIDTH
    g = generator._make_green_optics(h, w)
    r = generator._make_red_optics(h, w)
    t0=time.time()
    f = generator.render_frame(i, h, w, (g, r))
    return time.time()-t0

# sequential baseline
def _seq(i, g, r):
    h,w = config.HEIGHT, config.WIDTH
    t0=time.time(); f = generator.render_frame(i,h,w,(g,r)); return time.time()-t0

g = generator._make_green_optics(config.HEIGHT, config.WIDTH)
r = generator._make_red_optics(config.HEIGHT, config.WIDTH)
t0=time.time(); _seq(0,g,r)
print(f"sequential: {(time.time()-t0):.2f}s per frame")

for W in [2,4,8,16]:
    idx = [100+i for i in range(20)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=W) as ex:
        res = list(ex.map(_work, idx))
    per = (time.time()-t0)/len(idx)
    print(f"workers={W}: {per:.2f}s per frame  (avg {np.mean(res):.2f}s wall)")
