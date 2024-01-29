import numpy as np
#from scipy.sparse.linalg import gmres
from scipy.signal import convolve
from tests.cpp_tests.eigen.gmres.full_gmres.wrapper import PyScat
from Ncyl.cqm import scattering
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt

nmode = 10
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3, sum_num=nmode)

N = len(cyls)*(2*nmode + 1)

for i in range(len(cyls)):
    cyls[i].label = i

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

#N = len(cyls)*(2*nmode+1)
#np.random.seed(1)
#rhs = np.random.rand(N) + 1j*np.random.rand(N)
#rhs = rhs/np.linalg.norm(rhs)
tds.fzt_boundary_allcyls()
ks = tds.ks[:len(tds.rhsmat)]
print('len ks', len(ks))

sol_errs = []
mvp_err = []
#set up and solve the system
for i in range(len(tds.rhsmat)):
    rhs = tds.rhsmat[i]
    sc = scattering(cyls, rhs, k=ks[i], use_2cyl_modes=False)
    rhs = sc.rhs
    sc.make_root_matrix()
    sc.make_jhm()
    A = sc.make_scattering_blocks()
    print(np.amax(np.abs(A)))
    xact = np.linalg.solve(A, rhs)
    
    #test mvp
    ps = PyScat(cyls, ks[i], nmode) #create a PyScat instance
    ps.mvp_init() # make the fftw plans
    xout = ps.my_mvp(rhs) # perform 1 c++ matrix-free mvp
    rmat = A@rhs # 1 full mvp
    mvp_err += [np.max(np.abs(xout - rmat))]
    
    print()
    ps.gmres_setup(N, 0, 1e-10) #gmres initialization, args are (restart, maxiters, tol)
    xsol = ps.gmres(rhs) #call cython (and hence c++) gmres
    ps.fftw_clean_py() #free memory associated with fftw
    print('fftw cleaned up')
    sol_errs += [np.max(np.abs(xsol-xact))]

#%%
plt.loglog(np.abs(ks), mvp_err, label='mvp error')
plt.loglog(np.abs(ks), sol_errs, label='gmres error')
plt.legend()
plt.xlabel('|k|')
plt.ylabel('infty norm')
plt.savefig('/home/nick/scattering/tests/cpp_tests/eigen/gmres/full_gmres/errs_vs_k.pdf')
