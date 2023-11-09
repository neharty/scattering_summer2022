import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, scattering
import pickle
#%%
# np.random.seed(195)
# Rmax = 10
# radius = 0.05
# cyls = example_setups.random_circ_hex_vol_frac(Rmax, 'd', radius, 0.1)
# #%%
# print(len(cyls), len(cyls)*(radius**2)/Rmax**2)
# fig, ax = plot_cyls.plot_cyls(cyls)
# plt.show()

cyls = pickle.load(open('/home/nick/scattering/data/time_domain_tests/average_fl_waveform/cyls/bad/n/cyls.p', 'rb'))
fig, ax = plot_cyls(cyls)
plt.show()
