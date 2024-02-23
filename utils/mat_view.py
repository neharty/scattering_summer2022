import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import scattering
from utils import example_setups
from waveforms.waveforms import gaussian, gaussian_deriv

Nmode = 10
Nt=12
tint=None
T=40
wv=gaussian(time_shift=8,
            sigma=0.1)
X, Y = np.meshgrid([10], [0])
np.random.seed(1)
cyls = example_setups.uniform_grid_NxN(4, ['d', 'n'], 1, 3, sum_num=Nmode, bc_mode='uniform')
for i in range(len(cyls)):
    cyls[i].label = i
sc = scattering(cyls, np.zeros((2*Nmode+1)*len(cyls)), use_2cyl_modes=False, k=0.45j)
A = sc.make_scattering_blocks()

im = plt.matshow(np.abs(A))
plt.colorbar(im)

