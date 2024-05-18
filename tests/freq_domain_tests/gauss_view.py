import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5
from impedance import nguyen
from scipy.fft import fft, ifft, rfft, irfft, rfftfreq
import pickle
import argparse
import os

wv = gaussian(
        time_shift=12, 
        src_type='planewave',
        sigma=1.5, 
        inc_angle=np.pi)
T=80
dt = T/2**10
freqs = rfftfreq(2**10, d=dt)
k = 2*np.pi*freqs
ka = 2*np.pi*freqs*0.25
pw = np.abs(rfft(wv.profile(np.linspace(0, T, 2**10)-T//2)))

plt.hlines(1e-5, k[0], k[-1], 'k')
plt.loglog(k, pw)