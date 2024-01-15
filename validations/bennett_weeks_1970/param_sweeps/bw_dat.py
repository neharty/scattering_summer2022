import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from Ncyl.cqm import time_domain_scattering, cyl
from waveforms.waveforms import gaussian_deriv, gaussian
from utils import example_setups
from integrators.integrators import IRK3, IRK5
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('tint', help='time integration scheme')
parser.add_argument('Nt', type=int, help='time exponent')
parser.add_argument('Nmode', type=int, help='mode sum index')
parser.add_argument('bcs', help='boundary conds')
parser.add_argument('srctype', help='source type')
parser.add_argument('incang', type=int, help='incident angle')
parser.add_argument('radius', type=float, help='radius of the circles')
parser.add_argument('dspace', type=float, help='spacing multiplier of the circles')

args = parser.parse_args()

if args.incang == 1:
    incang = 0
elif args.incang == 2:
    incang = np.pi/6
elif args.incang == 3:
    incang = np.pi/4
elif args.incang == 4:
    incang = np.pi/3
elif args.incang == 5:
    incang = np.pi/2
        

#%%
#init variables
tau=1
n = 2/tau
T = 40

#point source
wv = gaussian(
            time_shift=4,
            sigma=1/n**2,
            amp = n/np.sqrt(np.pi),
            inc_angle=incang)

dpath = '/home/nick/scattering/validations/bennett_weeks_1970/param_sweeps/data/'

tint = args.tint

#%%
# 
a = args.radius
xpos = a*(1 + args.dspace/2)
cyls = [cyl(args.bcs[0], [-xpos, 0], a), cyl(args.bcs[1], [xpos, 0], a)]
tds = time_domain_scattering(
                             cyls,
                             wv,
                             T,
                             modes=args.Nmode,
                             sampnum=2**args.Nt,
                             timeint=tint
                             )

xs = np.array([-3*a, 0, 3*a])
ys = np.array([-2*a, 0, 2*a])

Usc = tds.make_U_sc(X, Y)
for a in range(len(thetas)):
    arr = [[tds.times[i], Usc[a, a, i]] for i in range(len(tds.times))]
    np.savetxt(fpath+str(thetas[a])+'.dat', arr)
