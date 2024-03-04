import numpy as np
from scipy.sparse.linalg import gmres
from scipy.signal import convolve
from tests.cython_tests.full_gmres.wrapper import PyScat
from Ncyl.cqm import scattering
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5

nmode=4
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3, sum_num=nmode)
pos = [c.pos[0] for c in cyls]
tshift = np.max(pos) + 4
wv=gaussian(time_shift=tshift)
X, Y = np.meshgrid([3*2], [0]) #dummy point


print('# of cyls', len(cyls))
N = len(cyls)*(2*nmode + 1)
for i in range(len(cyls)):
    cyls[i].label = i
Nt=12
tint=IRK5() #fix this for now
T=40
X, Y = np.meshgrid([3*2], [0])

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False
                             )
tds.fzt_boundary_allcyls_highorder()
ks = 1j*tds.evals/tds.dt
print(ks.shape)
#plot wavenums
for i in range(3):
    plt.plot(ks.real[:, i], ks.imag[:, i], label='c='+str(i))  
    plt.xlabel('real(k)')
    plt.ylabel('im(k)')
    plt.title('IRK5 wavenumbers')
plt.legend()
plt.savefig('/home/nick/scattering/proposal_imgs/s3/irk5_all_evals.pdf')
plt.show()

l = len(ks[: ,1])

plt.plot(ks.real[:l//2+1, 1], ks.imag[:l//2+1, 1],  'k-', label='c='+str(1))
plt.plot(ks.real[l//2+1:, 1], ks.imag[l//2+1:, 1],  'k-')  
plt.xlabel('real(k)')
plt.ylabel('im(k)')
plt.title('IRK5 solved wavenumbers')
plt.legend()
plt.savefig('/home/nick/scattering/proposal_imgs/s3/irk5_solved_evals.pdf')