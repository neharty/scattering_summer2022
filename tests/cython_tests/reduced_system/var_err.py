import numpy as np
from scipy.sparse.linalg import gmres
from scipy.signal import convolve
from Ncyl.scython_reduced.wrapper import PyScat
from Ncyl.cqm_reduced import scattering, time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt

nmode = 10
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3, sum_num=nmode)

N = len(cyls)*(2*nmode + 1)

Nt=12
tint=None
T=40
wv=gaussian(time_shift=6)
X, Y = np.meshgrid([6], [0])

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False
                             )
tds.fzt_boundary_allcyls()
ks = tds.ks[:len(tds.rhsmat)]
print('len ks', len(ks))

st_errs = []
jm_err = []
hm_err = []
#set up and solve the system
for i in range(len(tds.rhsmat)):
    rhs = tds.rhsmat[i]
    sc = scattering(cyls, rhs, k=ks[i], use_2cyl_modes=False, method='explicit', gmres_tol=1e-10)
    rhs = sc.rhs
    st_py = sc.make_root_matrix()
    jm_py = sc.make_jm()
    hm_py = sc.make_hm()
    
    ps = PyScat(cyls, ks[i], nmode) #create a PyScat instance
    st_cy = ps.make_root_matrix_py()
    jm_cy = ps.make_jm_py()
    hm_cy = ps.make_hm_py()
    if i == 0:
        sterrmat = np.abs(st_cy - st_py)
    st_errs += [np.max(np.abs(st_cy - st_py))]
    jmax = [np.max(np.abs(jm_cy[p] - jm_py[p])) for p in sc.labels]
    jm_err += [np.max(jmax)]
    
    hmax = [np.max(np.abs(hm_cy[p] - hm_py[p])) for p in sc.labels]
    hm_err += [np.max(hmax)]
    
#%%
plt.loglog(np.abs(ks), st_errs, label='st error')
plt.loglog(np.abs(ks), jm_err, label='jm error')
plt.loglog(np.abs(ks), hm_err, label='hm error')
plt.legend()
plt.xlabel('|k|')
plt.ylabel('infty norm')
plt.show()
#plt.savefig('/home/nick/scattering/tests/cpp_tests/eigen/gmres/full_gmres/errs_vs_k.pdf')
