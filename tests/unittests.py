import numpy as np
import unittest
import sys

sys.path.append('../Ncyl/')
from Ncyl import scattering, cyl

sc1 = scattering([cyl(0, 'd', pos=np.array([0,0]), radius = 0.5)], precond = 'simple')
sc2 = scattering([cyl(0, 'd', pos=np.array([0,0]), radius = 0.3), cyl (1, 'd', pos=np.array([0, 1.1]), radius = 0.3), cyl(2, 'd', pos=np.array([0, -1.1]), radius = 0.3)], precond = 'simple')
print(sc2.make_u(1,0))

from Ncyl_gmres import scattering, cyl
scgmres1 = scattering([cyl('d', np.array([0,0]), 0.5)])
scgmres2 = scattering([cyl('d', np.array([0,0]), 0.3), cyl ('d', np.array([0, 1.1]), 0.3), cyl('d', np.array([0, -1.1]), 0.3)])
print(scgmres2.make_u(1,0))

print(np.abs(sc1.make_u(1,0) - scgmres1.make_u(1,0)))
print(np.abs(sc2.make_u(1,0) - scgmres1.make_u(1,0)))
