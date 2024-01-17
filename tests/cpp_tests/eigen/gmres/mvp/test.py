import numpy as np
from scipy.sparse.linalg import gmres
import sys
from tests.cpp_tests.eigen.gmres.mvp.wrapper import my_mvp
from Ncyl.cqm import scattering as sc
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt

cyls = [cyl('d', [-1.5, 0], 1), cyl('d', [1.5, 0], 1)]

nmode = 8
N = len(cyls)*(2*nmode+1)
rhs = np.array(np.random.random(N), dtype=np.complex128)
print(my_mvp(cyls, nmode, 1, rhs))
