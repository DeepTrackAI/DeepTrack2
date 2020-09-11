import deeptrack.features as features
import numpy as np

import deeptrack as dt

for _ in range(4):
    A = features.DummyFeature(
        a=lambda: np.random.randint(10) * 1000,
    )
    B = features.DummyFeature(
        a=A.a,
        b=lambda a: a + np.random.randint(10) * 100,
    )
    C = features.DummyFeature(
        b=B.b,
        c=lambda b: b + np.random.randint(10) * 10,
    )
    D = features.DummyFeature(
        c=C.c,
        d=lambda c: c + np.random.randint(10) * 1,
    )
    AB = (A + (B + (C + D) ** 2) ** 2 ) ** 2

    for idx in range(100):
        

        output = AB.update().resolve(0)
        al = output.get_property("a", get_one=False)[::3]
        bl = output.get_property("b", get_one=False)[::3]
        cl = output.get_property("c", get_one=False)[::2]
        dl = output.get_property("d", get_one=False)
        print(idx)

                    
print(al)
print(bl)
print(cl)
print(dl)