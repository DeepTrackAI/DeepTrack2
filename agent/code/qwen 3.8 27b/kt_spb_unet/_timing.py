import time, warnings; warnings.filterwarnings("ignore")
import numpy as np, deeptrack as dt
def E(x,y,r):
    return dt.Ellipsoid(position=(y,x,0),position_unit="pixel",radius=r*1e-7,intensity=1.0)
def P(x,y,i=1.0):
    return dt.PointParticle(position=(y,x,0),position_unit="pixel",intensity=i)
def opt():
    return dt.Fluorescence(NA=1.2,resolution=1e-6,magnification=10,wavelength=525e-9,padding=(32,32,32,32),output_region=(0,0,416,640))
o = opt()

# cost: 1 ellipse
t0=time.time()
for _ in range(3):
    _ = np.asarray((o(E(300,200,14))).resolve(),dtype=np.float64)
print(f"1 ellipse: {(time.time()-t0)/3*1000:.0f} ms")

# cost: 6 ellipses chained
t0=time.time()
s=None
for k in range(6): s = E(300+k,200+k,14) if s is None else (s & E(300+k,200+k,14))
img=np.asarray((o(s)).resolve(),dtype=np.float64)
print(f"6 ellipses (build+image): {(time.time()-t0)*1000:.0f} ms")

# cost: 10 point particles
t0=time.time()
s=None
for k in range(10): s = P(10+k,20,1.0) if s is None else (s & P(10+k,20,1.0))
img=np.asarray((o(s)).resolve(),dtype=np.float64)
print(f"10 point particles: {(time.time()-t0)*1000:.0f} ms")
