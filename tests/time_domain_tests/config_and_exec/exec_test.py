import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5
from impedance import M12L02 as imp
import pickle
import ast

import argparse
import os
import configparser

parser = argparse.ArgumentParser()
parser.add_argument('cfg', help='config file location')
parser.add_argument('fout', help='full data output path')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.cfg)

#wave setup
if config['WAVE']['type'] == 'g':
    wv = gaussian(
            time_shift=float(config['WAVE']['time_shift']), 
            src_type=config['WAVE']['src_type'],
            sigma=float(config['WAVE']['sigma']),
            r0=ast.literal_eval(config['WAVE']['r0']))
elif config['WAVE']['type'] == 'dg':
    wv = gaussian_deriv(
            time_shift=float(config['WAVE']['time_shift']), 
            src_type=config['WAVE']['src_type'],
            sigma=float(config['WAVE']['sigma']),
            r0=ast.literal_eval(config['WAVE']['r0']))

# measurement points
thetas = np.arange(8)*np.pi/4
r = np.array([2.5, 5, 7.5, 10])
R, ANG = np.meshgrid(r, thetas)
X = R*np.cos(ANG)
Y = R*np.sin(ANG)


#cyls config
np.random.seed(float(config['CYLS']['seed']))
if config['CYLS']['geometry'] == 'circ':
    cyls = example_setups.random_circ_hex_vol_frac(
        float(config['CYLS']['R']), 
        'd', 
        float(config['CYLS']['a']),
        float(config['CYLS']['rho']),
        measurement_points=(X, Y)
        )
elif config['CYLS']['geometry'] == 'rect':
    cyls = example_setups.random_box_hex_vol_frac(
        float(config['CYLS']['Lx']), 
        float(config['CYLS']['Ly']), 
        bc, 
        float(config['CYLS']['a']), 
        float(config['CYLS']['rho'])
        )
print('num of cyls:', len(cyls))

#set up time domain sim
T = float(config['TDS']['T'])
tint

tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = int(config['TDS']['Nt']), 
                        modes=int(config['TDS']['Nmodes']),
                        timeint=tint,
                        balanced_system=config['TDS'].getboolean('balanced_system'),
                        use_par=config['TDS'].getboolean('use_par'),
                        use_cpp=config['TDS'].getboolean('use_cpp'),
                        scargs={'method':'explicit'}
                        )

t = tds.times
U = tds.make_U_sc(X, Y)

#write out the data
dataout = args.fout
if not os.path.exists(dataout):
    os.makedirs(dataout)

for ang in range(len(thetas)):
    for rdx in range(len(r)):
        fname = f'{r[rdx]:.3f}_{thetas[ang]:.3f}.dat'
        arr = [[t[i], U[ang, rdx, i]] for i in range(len(t))]
        np.savetxt(dataout+fname, arr)
