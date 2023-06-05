import numpy as np
from Ncyl.tds import scattering, cyl
from utils import example_setups
import matplotlib.pyplot as plt

N = 2
ks = np.logspace(-3, 2, num=100)
fs = ks/np.pi/2
bs = 2+10.0**np.arange(-4, 2)
for b in bs:
    ant = []
    roh = []
    for f in fs:
        cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, b)
        sc = scattering(cyls, freq=f)
        ant += [cyls[0].sum_index]
        md = sc.mode_select()
        roh += [md]
    np.savetxt('N_l_v_ka_antoine_'+str(b)+'.dat', np.array([ks.tolist(), ant]).T)
    np.savetxt('N_l_v_ka_rohfritsch_'+str(b)+'.dat', np.array([ks.tolist(), roh]).T)

