import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq

from Ncyl.tds import cyl, scattering, time_domain_scattering
from waveforms.waveforms import gaussian_deriv
from precond import preconditioner
from utils import example_setups

cyls = example_setups.uniform_grid_NxN(2, ['d', 'd', 'd', 'd'], 1, 3)

x = np.linspace(-5, 5, num=11)
y = [0]
T = 20
X, Y = np.meshgrid(x, y)

td = time_domain_scattering(
        cyls,
        gaussian_deriv(time_shift=8),
        T
        )

U = td.make_U_sc(X, Y)

M = U.max()
m = U.min()

for ypt in range(len(y)):
    for xpt in range(len(x)):
        plt.clf()
        plt.plot(td.times, U[ypt, xpt, :])
       
        plt.title('pos = ' + str((x[xpt], y[ypt])))
        plt.tight_layout()
        plt.savefig(str(str((x[xpt], y[ypt])))+'.png')

