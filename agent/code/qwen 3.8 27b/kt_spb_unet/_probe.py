import warnings; warnings.simplefilter("ignore")
import numpy as np, deeptrack as dt
H, W = 200, 400
def grab(f): return np.asarray(f.resolve(), dtype=np.float64).reshape(H, W)
def E(x, y, rm): return dt.Ellipsoid(position=(y,x,0), position_unit='pixel', radius=rm, intensity=1.0)
def P(x, y, i=1.0): return dt.PointParticle(position=(y,x,0), position_unit='pixel', intensity=i)
opt = dt.Fluorescence(NA=1.2, resolution=1e-6, magnification=10, wavelength=525e-9, padding=(32,32,32,32), output_region=(0,0,H,W))

print("== ellipse radius vs footprint (100 nm/px) ==")
for rm in [1e-7, 5e-7, 1.4e-6, 1.8e-6]:
    img = grab(opt(E(200,100,rm)) >> dt.NormalizeMinMax())
    m = img > 0.5*img.max()
    ys, xs = np.where(m)
    w = int(xs.max()-xs.min()) if len(xs) else 0
    print(f"radius={rm:.1e} m -> half-max width {w} px")

print("== point particle per-voxel max vs ellipse per-voxel max (same intensity=1.0) ==")
p = grab(opt(P(100,100,1.0)) >> dt.NormalizeMinMax())
c = grab(opt(E(100,100,1.4e-6)) >> dt.NormalizeMinMax())
print("point max:", round(float(p.max()),3), " ellipse max:", round(float(c.max()),3))
