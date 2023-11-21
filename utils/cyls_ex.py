import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, scattering
import pickle
from pympler.asizeof import asizeof
#%%
thetas = np.arange(8)*np.pi/4
r = np.array([2.5, 5, 7.5, 10])
R, ANG = np.meshgrid(r, thetas)
X = R*np.cos(ANG)
Y = R*np.sin(ANG)

np.random.seed(195)
Rmax = 10
radius = 0.05
cyls = example_setups.random_circ_hex_vol_frac(Rmax, 'd', radius, 0.2, measurement_points=(X, Y))
#%%
print(asizeof(cyls))
print(len(cyls), len(cyls)*(radius**2)/Rmax**2)
fig, ax = plot_cyls(cyls)
plt.show()

# cyls = pickle.load(open('/home/nick/scattering/data/time_domain_tests/average_fl_waveform/cyls/bad/n/cyls.p', 'rb'))
# fig, ax = plot_cyls(cyls)
# plt.show()
