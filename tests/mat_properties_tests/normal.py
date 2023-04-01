import numpy as np
from Ncyl.tds import cyl, scattering

rads = np.random.uniform(0.1, 0.4, size=3)
sc = scattering(
    [
        cyl('d', np.array([0,0]), rads[0]), 
        cyl ('d', np.array([0, 1.1]), rads[1]), 
        cyl('d', np.array([0, -1.1]), rads[2])
    ]
    )
A = sc.make_scattering_blocks()
Astar = A.T.conjugate()

print(np.allclose(A@Astar, -Astar@A))
