import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, scattering

#%%
thetas = np.arange(8)*np.pi/4
r = np.array([2.5, 5, 7.5, 10])
R, ANG = np.meshgrid(r, thetas)
X = R*np.cos(ANG)
Y = R*np.sin(ANG)

np.random.seed(100)
Rmax = 10
radius = 0.2
cyls = example_setups.random_circ_hex_vol_frac(
            Rmax, 
            'd', 
            radius, 
            0.3, 
            measurement_points=(X, Y)
        )
#%%
#check for overlap
cnrs = [c.pos for c in cyls]
truth = []
for p in cnrs:
    dist = np.sqrt((X - p[0])**2 + (Y - p[1])**2) 
    mask = dist > radius
    truth += [np.all(mask)]
#print([c.pos for c in cyls[mask]])
print(np.all(truth))
#%%
print(len(cyls), len(cyls)*(radius**2)/Rmax**2)
fig, ax = plot_cyls(cyls)
ax.plot(X, Y, 'kx')
plt.show()

# cyls = pickle.load(open('/home/nick/scattering/data/time_domain_tests/average_fl_waveform/cyls/bad/n/cyls.p', 'rb'))
# fig, ax = plot_cyls(cyls)
# plt.show()
