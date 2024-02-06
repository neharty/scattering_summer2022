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

np.random.seed(1)
Rmax = 10
radius = 0.3
vfill = 0.3
minspace = radius*1
# cyls = example_setups.random_circ_hex_vol_frac(
#             Rmax,
#             ['d', 'n'], 
#             radius, 
#             vfill,
#             minspace=minspace,
#             impedance=1,
#             bc_mode='uniform',
#             rin=0
#         )
Lx = 5
Lxin = 2
Ly = 10
Lyin = 5
cyls = example_setups.random_box_hex_vol_frac(
        5, 
        10, 
        ['d', 'n'], 
        radius, 
        vfill, 
        bc_mode='uniform', 
        minspace=minspace, 
        Lxin = Lxin,
        Lyin = Lyin
        )
#%%
#check for overlap with measurement points
cnrs = [c.pos for c in cyls]
truth = []
for p in cnrs:
    dist = np.sqrt((X - p[0])**2 + (Y - p[1])**2) 
    mask = dist > radius
    truth += [np.all(mask)]
#print([c.pos for c in cyls[mask]])
print(np.all(truth))

#check if they meet the minimum distance
ismindist = True
for p in cnrs:
    for q in cnrs:
        tmp = np.linalg.norm(p-q)
        if tmp <= (2*radius + minspace) and tmp > 0:
            ismindist = False
            #print(p, q)
            break
print(ismindist)

#%%

fig, ax = plot_cyls(cyls)
#ax.plot(X, Y, 'kx')
plt.show()
print(len(cyls), len(cyls)*(radius**2)*np.pi/(4*(Lx-Lxin)*(Ly-Lyin)))#, len(cyls)*(radius**2)/Rmax**2)

# cyls = pickle.load(open('/home/nick/scattering/data/time_domain_tests/average_fl_waveform/cyls/bad/n/cyls.p', 'rb'))
# fig, ax = plot_cyls(cyls)
# plt.show()
