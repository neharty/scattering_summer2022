import numpy as np
import os
import argparse

from Ncyl.cqm import cyl, time_domain_scattering
from utils import example_setups
from waveforms.waveforms import gaussian

parser=argparse.ArgumentParser()
parser.add_argument('Nsamp', type=int)
parser.add_argument('Nmode', type=int)
parser.add_argument('bc')
parser.add_argument('radidx', type=int)
args = parser.parse_args()

Nsamp = args.Nsamp #power of the sample number
Nmode = args.Nmode #number of modes
bc= args.bc
radidx = args.radidx

dataout = '/home/nick/scattering/data/cqm/td_fl/rad_dep'
os.makedirs(dataout, exist_ok=True)

Nsamp = 2**Nsamp
rads = 10.**np.array([-6, -5, -4, -3, -2, -1])

rad = rads[radidx]
cyls = example_setups.big_circle(20, bc, 1, rad)
wvf = gaussian(inc_angle=np.pi/2, time_shift=2, sigma=0.01)
    
td = time_domain_scattering(
            cyls, 
            wvf, 
            8, 
            sampnum=Nsamp, 
            modes=Nmode,
            scargs={'method':'explicit'}
            )
X, Y = np.meshgrid([0], [0])
U = td.make_U_sc(X, Y)
darr = [[td.times[i], -U[0,0,i]] for i in range(len(td.times))]
np.savetxt(os.path.join(dataout, str(rad)+'_'+bc+'.dat'), darr)