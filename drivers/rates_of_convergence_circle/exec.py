import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gaussian_norm_width, gaussian_deriv
from utils import example_setups
from integrators.integrators import IRK3, IRK5
import impedance
import pickle
import ast
from time import time
import logging

import argparse
import os
import configparser

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int, help='random number seed')
parser.add_argument('volfrac', type=float, help='target volume fraction')
parser.add_argument('width', help='width of gaussian pulse', type=float)
parser.add_argument('cfg', help='config file location')
parser.add_argument('fout', help='full data output path')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.cfg)

#make data output folder
dataout = os.path.join(
        args.fout,
        config['CYLS']['geometry'],
        'volfrac_'+str(args.volfrac),
        'pulsewidth_'+str(args.width),
        'seed_' + str(args.seed)
        )
if not os.path.exists(dataout):
    os.makedirs(dataout)

#debug output file
log = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(dataout, 'debug.log'), level=logging.DEBUG, filemode='w')

# wave setup
print('generating wave')
if config['WAVE']['type'] == 'g':
    wv = gaussian_norm_width(
            src_type=config['WAVE']['src_type'],
            width=args.width,
            r0=ast.literal_eval(config['WAVE']['r0']))
elif config['WAVE']['type'] == 'dg':
    wv = gaussian_deriv(
            src_type=config['WAVE']['src_type'],
            sigma=args.width,
            r0=ast.literal_eval(config['WAVE']['r0']))

# measurement points
thetas = np.arange(8)*np.pi/4
r = np.array([11, 12, 13, 14, 15])
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

np.random.seed(args.seed)
if config['CYLS']['geometry'] == 'circ':
    #big circle
    rout = float(config['CYLS']['rout'])
    a = float(config['CYLS']['a'])
    rho = args.volfrac
    rin = float(config['CYLS']['rin'])
    cyls = example_setups.random_circ_hex_vol_frac(
        rout, 
        ast.literal_eval(config['CYLS']['bcs']), 
        a,
        rho,
        minspace = a*float(config['CYLS']['dmin']), 
        measurement_points=(X, Y), 
        rin=rin, 
        impedance=imp,
        bc_mode=config['CYLS']['bc_mode']
        )
    log.info('actual percentage fill is %f', 
            len(cyls)*a**2/(rout**2 - rin**2))

elif config['CYLS']['geometry'] == 'rect':
    #rectangle
    Lx = float(config['CYLS']['Lx'])
    Ly = float(config['CYLS']['Ly'])
    a = float(config['CYLS']['a'])
    rho = args.volfrac
    Lxin = float(config['CYLS']['Lxin'])
    Lyin = float(config['CYLS']['Lyin'])
    cyls = example_setups.random_box_hex_vol_frac(
        Lx, 
        Ly, 
        ast.literal_eval(config['CYLS']['bcs']), 
        a, 
        rho,
        minspace = float(config['CYLS']['a'])*float(config['CYLS']['dmin']), 
        measurement_points=(X, Y),
        Lxin=Lxin,
        Lyin=Lyin,
        impedance=imp,
        bc_mode=config['CYLS']['bc_mode']
        )
    log.info('actual percentage fill is %f', 
            len(cyls)*a**2*np.pi/(4*(Lx*Ly - Lxin*Lyin)))

print('num of cyls:', len(cyls))
log.info('number of cyls is %d', len(cyls))

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
solvetime = time()
tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = 2**int(config['TDS']['Nt']), 
                        timeint=tint,
                        mode_tol=float(config['TDS']['mode_tol']),
                        use_par=config['TDS'].getboolean('use_par'),
                        use_cython=config['TDS'].getboolean('use_cython'),
                        cpus=int(config['TDS']['cpus']),
                        scargs={'method':'gmres'}
                        )

t = tds.times
U = tds.make_U_sc(X, Y)
solvetime = time() - solvetime
log.info('this run took %f seconds', solvetime)
log.info('used 2*%d+1 modes for each cyl', tds.modes)
print('time domain sim done')
print('writing data')

#write out the data
for ang in range(len(thetas)):
    for rdx in range(len(r)):
        fname = f'{r[rdx]:.3f}_{thetas[ang]:.3f}.dat'
        arr = [[t[i], U[ang, rdx, i]] for i in range(len(t))]
        np.savetxt(os.path.join(dataout, fname), arr)

#copy the config file to the data output directory
os.system('cp '+args.cfg+' '+os.path.join(dataout, 'config_copy.cfg'))

#write the circle data to the data output directory
pickle.dump(cyls, open(os.path.join(dataout, "cyls.p"), "wb"))

print('done')
