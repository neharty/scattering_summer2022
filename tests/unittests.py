import numpy as np
import unittest
import sys
sys.path.append('../Ncyl/')
from Ncyl import scattering, cyl

sc1 = scattering([cyl(0, 'd', pos=np.array([0,0]), radius = 0.5)])
sc2 = scattering([cyl(0, 'd', pos=np.array([0,0]), radius = 0.5), cyl (1, 'n', pos=np.array([0, 1.1]), radius = 0.5), cyl(2, 'i', pos=np.array([0, -1.1]), radius = 0.5)])
sc2.make_u(1,0)
