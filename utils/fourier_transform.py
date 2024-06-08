import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5
from impedance import nguyen
from scipy.fft import fft, ifft, rfft, irfft, rfftfreq
import pickle
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('ttrace_path', help = 'path to time trace data file')
parser.add_argument('radius', help = 'cylinder radius', type=float)
parser.add_argument('sigma', help='gaussian width parameter', type=float)
parser.add_argument('fout', help='full data output path')

args = parser.parse_args()

Utmp = np.loadtxt(args.ttrace_path)
times = Utmp[:, 0]
U = Utmp[:, 1]
NNt = len(U)-1
dt = times[-1]/NNt
freqs = rfftfreq(NNt, d=dt)
ka = 2*np.pi*freqs*args.radius

#find the incident planewave amplitude
wv = gaussian(sigma=args.sigma)
pw = np.abs(rfft(wv.profile(np.linspace(0, times[-1], NNt)-times[-1]//2)))
freqerr = 1e-5
pwmask = (pw > freqerr) #
ka = ka[pwmask]

#transform the time domain data
freqdat = np.abs(rfft(U[:-1]))
freqdat = freqdat[pwmask]
print(len(ka), freqdat.shape)
for i in range(len(ka)):
    if pw[i] > freqerr:
        freqdat[i] /= pw[i]
    else:
        freqdat[:, i] = 0
    #freqdat[:, i] /= np.max(np.abs(freqdat[:, i]))

# save freq domain data
head, tail = os.path.split(args.ttrace_path)
fnametmp, tmp = os.path.splitext(tail)
fname = fnametmp + '_ka.dat'
outparr = np.zeros((len(ka), 2))
outparr[:, 0] = ka[:]
outparr[:, 1] = freqdat[:]
os.makedirs(args.fout, exist_ok=True)
np.savetxt(os.path.join(args.fout , fname), outparr)