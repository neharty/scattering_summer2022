import numpy as np
from scipy.signal import convolve
from Ncyl.cqm_reduced import time_domain_scattering, cyl
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from time import process_time, time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5
import pstats, cProfile

Nmode = 5
Nt=12
tint=None
T=40
wv=gaussian(time_shift=12)
X, Y = np.meshgrid([10], [0])
cyls = example_setups.uniform_grid_NxN(5, 'd', 1, 3, sum_num=Nmode)

#%%
#with cpp
def init():
    tdscpp = time_domain_scattering(
                                 cyls,
                                 wv,
                                 T,
                                 modes=Nmode,
                                 sampnum=2**Nt,
                                 timeint=tint,
                                 use_par=False,
                                 scargs={'method':'gmres', 'gmres_tol':1e-10},
                                 use_cpp=True
                                 )
    return tdscpp

def run(tdscpp):
    Usccpp = tdscpp.make_U_sc(X, Y)
