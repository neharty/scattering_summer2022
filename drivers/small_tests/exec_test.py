import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian, gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5
import impedance
import pickle
import ast

import argparse
import os
import configparser

parser = argparse.ArgumentParser()
parser.add_argument('seed', dtype=int, help='random number seed')
parser.add_argument('cfg', help='config file location')
parser.add_argument('fout', help='full data output path')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.cfg)

# wave setup
print('generating wave')
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
print('generating cylinders')
tmpimp = config['CYLS']['impedance']
if tmpimp == 'none':
    imp=None
else:
    imp = getattr(impedance, config['CYLS']['impedance'])

np.random.seed(int(config['CYLS']['seed']))
if config['CYLS']['geometry'] == 'circ':
    #big circle
    cyls = example_setups.random_circ_hex_vol_frac(
        float(config['CYLS']['rout']), 
        ast.literal_eval(config['CYLS']['bcs']), 
        float(config['CYLS']['a']),
        float(config['CYLS']['rho']),
        minspace = float(config['CYLS']['a'])*float(config['CYLS']['dmin']), 
        measurement_points=(X, Y), 
        rin=float(config['CYLS']['rin']), 
        impedance=imp,
        bc_mode=config['CYLS']['bc_mode']
        )
elif config['CYLS']['geometry'] == 'rect':
    #little circle
    cyls = example_setups.random_box_hex_vol_frac(
        float(config['CYLS']['Lx']), 
        float(config['CYLS']['Ly']), 
        ast.literal_eval(config['CYLS']['bcs']), 
        float(config['CYLS']['a']), 
        float(config['CYLS']['rho']),
        minspace = float(config['CYLS']['a'])*float(config['CYLS']['dmin']), 
        measurement_points=(X, Y),
        Lxin=float(config['CYLS']['Lxin']),
        Lyin=float(config['CYLS']['Lyin']),
        impedance=imp,
        bc_mode=config['CYLS']['bc_mode']
        )
print('num of cyls:', len(cyls))

#set up time domain sim
T = float(config['TDS']['T'])
tmptint  = config['TDS']['tint']
if tmptint == 'bdf2':
    tint = None
elif tmptint == 'irk3':
    tint = IRK3()
elif tmptint == 'irk5':
    tint = IRK5()
print('setting up time domain simulation')
tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = 2**int(config['TDS']['Nt']), 
                        modes=int(config['TDS']['Nmodes']),
                        timeint=tint,
                        balanced_system=config['TDS'].getboolean('balanced_system'),
                        use_par=config['TDS'].getboolean('use_par'),
                        use_cython=config['TDS'].getboolean('use_cython'),
                        scargs={'method':'gmres'}
                        )

t = tds.times
U = tds.make_U_sc(X, Y)
print('time domain sim done')
print('writing data')

#write out the data
dataout = os.path.join(args.fout, args.seed)
if not os.path.exists(dataout):
    os.makedirs(dataout)

for ang in range(len(thetas)):
    for rdx in range(len(r)):
        fname = f'{r[rdx]:.3f}_{thetas[ang]:.3f}.dat'
        arr = [[t[i], U[ang, rdx, i]] for i in range(len(t))]
        np.savetxt(dataout+fname, arr)

#copy the config file to the data output directory
os.system('cp '+args.cfg+' '+os.path.join(dataout, 'config_copy.cfg'))

#write the circle data to the data output directory
pickle.dump(cyls, open(os.path.join(dataout, "cyls.p"), "wb"))