import numpy as np
from scipy.sparse.linalg import gmres
from scipy.signal import convolve
from tests.cython_tests.full_gmres.wrapper import PyScat
from Ncyl.cqm import scattering
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_norm_width
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt
from integrators.integrators import IRK3, IRK5

nmode=4
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3, sum_num=nmode)
pos = [c.pos[0] for c in cyls]
wv = gaussian_norm_width(width = 1, lag = 1)
np.random.seed(1234)
cyls = example_setups.random_circ_hex_vol_frac(10, 'd', 0.5, 0.2, minspace=0.125, rin=5)
X, Y = np.meshgrid([13], [0]) #dummy point
print('# of cyls', len(cyls))
N = len(cyls)*(2*nmode + 1)
for i in range(len(cyls)):
    cyls[i].label = i
Nt=15
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
ksdt = 1j*tds.evals
tds.set_timeshift()
tds.fzt_boundary_allcyls_highorder()
ks = 1j*tds.evals/tds.dt
print(ks.shape)
#%%
#plot wavenums
for i in range(3):
    if i == 0:
        plt.semilogy(ksdt.real[:, i], ksdt.imag[:, i], 'r--', label='c='+str(i))
    if i == 1:
        l = len(ksdt[:, i])
        plt.semilogy(ksdt.real[:l//2+1, i], ksdt.imag[:l//2+1, i], 'k', label='c='+str(i))
        plt.semilogy(ksdt.real[l//2+1:, i], ksdt.imag[l//2+1:, i], 'k')
    if i==2:
        l = len(ksdt[:, i])
        plt.semilogy(ksdt.real[:l//2+1, i], ksdt.imag[:l//2+1, i], 'b-.', label='c='+str(i))
        plt.semilogy(ksdt.real[l//2+1:, i], ksdt.imag[l//2+1:, i], 'b-.')
    plt.xlabel('real(k)')
    plt.ylabel('im(k)')
    plt.title(r'IRK5 wavenumbers, scaled by $\Delta t$')
plt.legend()
plt.tight_layout()
plt.savefig('/home/nick/scattering/proposal_imgs/s3/irk5_all_evals.pdf')
plt.show()

ks = ks[:tds.rhs_maxidx, 1]
#%%
plt.plot(ks.real, ks.imag,  'k.')
plt.xlabel('real(k)')
plt.ylabel('im(k)')
plt.title('IRK5 solved wavenumbers')
plt.tight_layout()
plt.savefig('/home/nick/scattering/proposal_imgs/s3/irk5_solved_evals.pdf')