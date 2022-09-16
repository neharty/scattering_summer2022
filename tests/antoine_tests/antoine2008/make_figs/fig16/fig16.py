import numpy as np
import matplotlib.pyplot as plt
from time import process_time

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl_gmres import scattering, cyl

import sys
sys.path.append('/home/n/nh287/scattering_summer2022/utils/')
import example_setups

print('fig16 start')

#dirichlet
freqs = np.linspace(1, 200, num=30)
mvp = np.zeros(len(freqs))

for rstrt = [1, 50]:
    for i in range(len(freqs)):
        sc = scattering(example_setups.uniform_grating(75, ['d' for i in range(75)], 1, 3), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10, gmres_restart=rstrt)
        
        sc.scattering_coeffs()
        mvp[i] = sc.get_mvp_counter()
        print(mvp[i], sc.get_mvp_counter())
        
        lbl = ''
        if rstrt == 1:
            lbl = 'precond GMRES'
        else:
            lbl = 'precond GMRES(50)'

    plt.plot(freqs, mvp, '.-', label = lbl)

plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('dirichlet bcs')
#plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig16/fig16_dirichlet.png')
plt.clf()

#neumann
for rstrt = [1, 50]:
    for i in range(len(freqs)):
        sc = scattering(example_setups.uniform_grating(75, ['d' for i in range(75)], 1, 3), freq=freqs[i]/(2*np.pi), sum_tol=1e-10, gmres_tol=1e-10, gmres_restart=rstrt)
        
        sc.scattering_coeffs()
        mvp[i] = sc.get_mvp_counter()
        print(mvp[i], sc.get_mvp_counter())
        
        lbl = ''
        if rstrt == 1:
            lbl = 'precond GMRES'
        else:
            lbl = 'precond GMRES(50)'

    plt.plot(freqs, mvp, '.-', label = lbl)

plt.plot(freqs, mvp, '.-')
plt.xlabel('ka')
plt.ylabel('MVP')
plt.title('neumann bcs')
#plt.legend()
plt.tight_layout()
plt.savefig('/home/n/nh287/scattering_summer2022/tests/antoine_tests/antoine2008/figs/fig16/fig16_neumann.png')
plt.clf()


