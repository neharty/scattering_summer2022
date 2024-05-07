import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
#from Ncyl.cqm import time_domain_scattering, cyl
from Ncyl.tds import scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls
parser = argparse.ArgumentParser()

parser.add_argument('seed', help='random number gen seed', type=int)
parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('Nmodes', help = 'number theta modes', type=int)
parser.add_argument('T', help = 'final computation time', type=float)
parser.add_argument('rho', help='volume fill percentage', type=float)
parser.add_argument('R', help='radius of scatterer-containing disk', type=float)
parser.add_argument('radius', help = 'cylinder scatterer radius', type=float)
parser.add_argument('dspace', help='minimum spacing parameter', type=float)
parser.add_argument('width', help='gaussian width parameter', type=float)
parser.add_argument('k', help='wavenumber', type=float)
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

#initialize problem
#figure 5
#length units are in cm here'
dists = [3*i for i in range(8)]
cyls = [cyl('n', [d, 0], 1) for d in dists]

fig, ax = plot_cyls(cyls)
ax.plot()
plt.show()

k=np.pi*0.4

#%%
yvals = np.linspace(-12, 12, num=1000)
sc = scattering(cyls, freq = k/(2*np.pi), use_2cyl_modes=False, sum_index=12, inc_angle=np.pi/2, method='explicit')
X, Y = np.meshgrid([0], yvals)
uy = sc.make_u(X, Y)

#%%
#y-axis solution
trace = np.abs(uy[:, 0])
plt.plot(yvals, trace, label='ka='+str(k))
plt.legend()
plt.xlabel('y [m]')
plt.ylabel('normed amp')
plt.xlim(-12, 12)
plt.ylim(0, 2)
plt.grid()
#plt.savefig('/home/nick/scattering/validations/rohfritsch_2019/fig5a_y.pdf')
plt.show()

arr = [[yvals[i], trace[i]] for i in range(len(yvals))]
#np.savetxt('/home/nick/scattering/validations/carerra_2008/dirichlet.dat', arr)

