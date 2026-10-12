import time, warnings; warnings.filterwarnings("ignore")
os_env = {"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
          "MKL_NUM_THREADS":"1","VECLIB_MAXIMUM_THREADS":"1",
          "NUMEXPR_NUM_THREADS":"1","OMP_DYNAMIC":"false"}
import os
for k,v in os_env.items(): os.environ[k]=v

import numpy as np, sys
sys.path.insert(0,"/a/kt_spb_unet")
import config, generator
from concurrent.futures import ProcessPoolExecutor

def init():
    generator._make_green_optics(config.HEIGHT, config.WIDTH)
    generator._make_red_optics(config.HEIGHT, config.WIDTH)

def work(i):
    h,w=config.HEIGHT,config.WIDTH
    t0=time.time()
    f=generator.render_frame(i,h,w,(init.g_opt,init.r_opt))
    return time.time()-t0
init.g_opt = init.r_opt = None
def init2():
    init.g_opt = generator._make_green_optics(config.HEIGHT, config.WIDTH)
    init.r_opt = generator._make_red_optics(config.HEIGHT, config.WIDTH)

N=48
for W in [4, 8, 16, 32]:
    idx=list(range(500,500+N))
    t0=time.time()
    with ProcessPoolExecutor(max_workers=W, initializer=init2) as ex:
        list(ex.map(work, idx))
    wall=(time.time()-t0)
    print(f"workers={W:2d}: wall {wall:7.1f}s  throughput {N/wall:4.2f} f/s  total {N} frames")
