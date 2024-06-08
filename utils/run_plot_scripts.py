import numpy as np
from utils import example_setups
from utils.plot_cyls import plot_cyls, plot_cyls_pub
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, scattering

#%%
# thetas = np.arange(8)*np.pi/4
# r = np.array([2.5, 5, 7.5, 10])
# R, ANG = np.meshgrid(r, thetas)
# X = R*np.cos(ANG)
# Y = R*np.sin(ANG)

# np.random.seed(1)
# Rmax = 10
# radius = 0.5
# vfill = 0.3
# minspace = radius*0.125
# # cyls = example_setups.random_circ_hex_vol_frac(
# #             Rmax,
# #             ['d', 'n'], 
# #             radius, 
# #             vfill,
# #             minspace=minspace,
# #             impedance=1,
# #             bc_mode='uniform',
# #             rin=0
# #         )
# Lx = 10
# Lxin = 5
# Ly = 10
# Lyin = 5
# cyls = example_setups.random_box_hex_vol_frac(
#         10, 
#         10, 
#         'd', 
#         radius, 
#         vfill, 
#         minspace=minspace, 
#         Lxin = Lxin,
#         Lyin = Lyin
#         )

# #cyls = example_setups.uniform_grid_NxN(3, ['d', 'n', 'i'], 1, 3, impedance=1, bc_mode='repeated')
# #%%
# #check for overlap with measurement points
# cnrs = [c.pos for c in cyls]
# truth = []
# for p in cnrs:
#     dist = np.sqrt((X - p[0])**2 + (Y - p[1])**2) 
#     mask = dist > radius
#     truth += [np.all(mask)]
# #print([c.pos for c in cyls[mask]])
# print(np.all(truth))

# #check if they meet the minimum distance
# ismindist = True
# for p in cnrs:
#     for q in cnrs:
#         tmp = np.linalg.norm(p-q)
#         if tmp <= (2*radius + minspace) and tmp > 0:
#             ismindist = False
#             #print(p, q)
#             break
# print(ismindist)

# #%%

# fig, ax = plot_cyls_pub(cyls)
# #ax.plot(X, Y, 'kx')
# plt.show()
# print(len(cyls), len(cyls)*(radius**2)*np.pi/(4*(Lx*Ly-Lxin*Lyin)))#, len(cyls)*(radius**2)/Rmax**2)

# # cyls = pickle.load(open('/home/nick/scattering/data/time_domain_tests/average_fl_waveform/cyls/bad/n/cyls.p', 'rb'))
# # fig, ax = plot_cyls(cyls)
# # plt.show()

cyls = example_setups.uniform_grid_NxN(3, 'd', 1, 3)
pos = [c.pos for c in cyls]

from scatterers import cyl, starfish, bean
from utils.plot_scatterers import plot_scatterers

rangs = np.random.uniform(0, 2*np.pi, len(cyls))
print(rangs)
scrs = [bean('d', pos[i], 1, -np.pi/4) for i in range(len(cyls))]
fig, ax = plot_scatterers(scrs)
#plt.savefig('/home/nick/3x3_starfish_geometry.pdf')
plt.show()