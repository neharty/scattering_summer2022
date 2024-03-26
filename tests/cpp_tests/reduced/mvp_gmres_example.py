import numpy as np
#from scipy.sparse.linalg import gmres
from scipy.signal import convolve
from tests.cpp_tests.reduced.wrapper import PyScat
from Ncyl.cqm_reduced import scattering, time_domain_scattering, cyl
from waveforms.waveforms import gaussian
from utils import example_setups
from time import process_time
import matplotlib.pyplot as plt

nmode = 10
cyls = example_setups.uniform_grid_NxN(2, 'd', 1, 3, sum_num=nmode)

Nmode = 10
Nt=12
tint=None
T=40
wv=gaussian(time_shift=6)
X, Y = np.meshgrid([6], [0])

tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=Nmode,
                             sampnum=2**Nt,
                             timeint=tint,
                             use_par=False
                             )
for i in range(len(cyls)):
    cyls[i].label = i

tds.fzt_boundary_allcyls()
ks = tds.ks
print('len ks', len(ks))

errs = []
#set up and solve the system
for i in range(len(tds.rhsmat)):
    rhs = tds.rhsmat[i]
    sc = scattering(cyls, rhs, k=ks[i], use_2cyl_modes=False)
    rhs = sc.rhs
    sc.make_root_matrix()
    sc.make_jm()
    sc.make_hm()
    A = sc.make_scattering_blocks()
    print('cond num', np.linalg.cond(A))
    xact = np.linalg.solve(A, rhs)
    print('xact computed')

    #test convolve

    #test mvp
    ps = PyScat(cyls, ks[i], nmode) #create a PyScat instance
    print('pyscat initialized')
    ps.mvp_init() # make the fftw plans
    print('mvp initialized')
    xout = ps.my_mvp(rhs) # perform 1 c++ matrix-free mvp
    print('1 mvp performed')
    rmvp = sc.mvp(rhs) # 1 python matrix-free mvp 
    rmat = A@rhs # 1 full mvp
    idx = np.argmax(np.abs(xout - rmat))
    print('mvp err', np.max(np.abs(xout - rmvp)))
    print('py mvp err', np.max(np.abs(rmat - rmvp)))
    print('mat mulerr', np.max(np.abs(xout - rmat)))
    print('my mvp', xout[idx])
    print('actual', rmat[idx])
    #xdiff = np.abs(xout-xact)
    
    print()
    xsol = ps.gmres_py(rhs, len(cyls)*(2*nmode+1), 1e-10) #call cython (and hence c++) gmres
    mvpsol = ps.my_mvp(xsol) #use c++ mvp with to approximate the rhs, then measure errors below
    print('gmres rel err w/mvp', np.max(np.abs(rhs - mvpsol))/np.max(np.abs(rhs)))
    print('gmres mat err', np.max(np.abs(rhs-(A@xsol)))/np.max(np.abs(rhs)))
    print('gmres backslash err', np.max(np.abs(xact-xsol))/np.max(np.abs(rhs)))
    #ps.fftw_clean_py() #free memory associated with fftw
    print('fftw cleaned up')
    
    input()
