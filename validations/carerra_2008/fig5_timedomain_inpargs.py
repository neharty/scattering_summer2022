import numpy as np
import matplotlib.pyplot as plt
from utils import example_setups
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_norm_width, switched_sinewave, gauss_mod_sine
from integrators.integrators import IRK5
import os
from scipy.fft import rfft, irfft, rfftfreq
from utils.plot_cyls import plot_cyls
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('Nt', help = 'number of time samples', type=int)
parser.add_argument('Nmodes', help = 'number theta modes', type=int)
parser.add_argument('T', help = 'final computation time', type=float)
parser.add_argument('width', help='gaussian width parameter', type=float)
parser.add_argument('ka_cutoff', help='cutoff k value for data output', type=float)
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

#initialize problem
#figure 5
wv = gaussian_norm_width(time_shift=0, inc_angle=-np.pi/2, width=args.width, lag=1)

T = args.T
tint = IRK5()
#length units are in cm here'
dists = [3*i for i in range(8)]
cyls = [cyl('n', [d, 0], 1) for d in dists]
#%%
yvals = np.linspace(-12, 12, num=1000)
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             sampnum=2**args.Nt,
                             modes=args.Nmodes,
                             timeint=tint,
                             use_cython=False,
                             use_par=True,
                             scargs={'method':'gmres'}
                             )
X, Y = np.meshgrid([0], yvals)
Uy = tds.make_U(X, Y)

#%%
freqs = rfftfreq(Uy.shape[-1]-1, d=tds.dt)
ka = 2*np.pi*freqs

#find the incident planewave amplitude
pw = rfft(wv.profile(np.linspace(0, T, 2**args.Nt)-T//2))

#write out the data
if not os.path.exists(args.fout):
    os.makedirs(args.fout)

for kidx in range(len(ka)):
    if ka[kidx] < args.ka_cutoff:
        fname = f'Nt{args.Nt}_Nmode{args.Nmodes}_T{args.T}_width{args.width}_ka{ka[kidx]:.3f}_td2freq.dat'
        freqdat = rfft(Uy[..., :-1])
        ff = freqdat[:, 0, kidx]
        amp = pw[kidx]
        trace = np.abs(ff)/np.abs(amp)
        arr = [[yvals[i], trace[i]] for i in range(len(yvals))]
        np.savetxt(os.path.join(args.fout , fname), arr)

