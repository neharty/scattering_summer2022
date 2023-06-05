import numpy as np
from Ncyl.tds import scattering, cyl
from utils import example_setups
import matplotlib.pyplot as plt

N = 2
ks = np.array([1e-2, 1, 1e2])
fs = ks/np.pi/2
bs = np.logspace(-6, 0, num=100)

for f in fs:
    ant = []
    roh = []
    for b in bs:
        cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, b+2)
        sc = scattering(cyls, freq=f, use_2cyl_modes=False)
        ant += [cyls[0].sum_index]
        md = sc.mode_select()
        roh += [md]
    np.savetxt('data/Nl_v_b/N_l_v_b_antoine_'+str(2*np.pi*f)+'.dat', np.array([bs.tolist(), ant]).T)
    np.savetxt('data/Nl_v_b/N_l_v_b_rohfritsch_'+str(2*np.pi*f)+'.dat', np.array([bs.tolist(), roh]).T)

