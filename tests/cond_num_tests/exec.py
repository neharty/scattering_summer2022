import numpy as np
from Ncyl.cqm import cyl, time_domain_scattering, scattering
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

runtime = time()

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int, help='random number seed')
parser.add_argument('volfrac', type=float, help='target volume fraction')
parser.add_argument('Nmode', help='width of gaussian pulse', type=int)
parser.add_argument('cfg', help='config file location')
parser.add_argument('fout', help='full data output path')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.cfg)

#cylinder string
ctype = config['CYLS']['geometry']
if ctype == 'circ':
    rin = float(config['CYLS']['rin'])
    if rin > 0:
        cylstr = 'donut'
    else:
        cylstr = 'filled_circ'
if ctype == 'rect':
    Lx = float(config['CYLS']['Lx'])
    Ly = float(config['CYLS']['Ly'])
    if Lx > 0 or Ly > 0:
        cylstr = 'rect_donut'
    else:
        cylstr = 'filled_rect'

#make data output folder
dataout = os.path.join(
        args.fout,
        cylstr,
        'volfrac_'+str(args.volfrac),
        'Nmode_'+str(args.Nmode),
        'seed_' + str(args.seed)
        )

if not os.path.exists(dataout):
    os.makedirs(dataout)

#debug output file
log = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(dataout, 'debug.log'), level=logging.DEBUG, filemode='w')

# wave setup
print('generating wave')
wv = gaussian_norm_width() #dummy wave

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
tint = IRK5()
log.info('using IRK5')
print('setting up time domain simulation')

tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = 2**int(config['TDS']['Nt']), 
                        timeint=tint,
                        modes = args.Nmode
                        )
condnum0 = []
condnum1 = []
condnum2 = []
for i in range(3):
    ks = 1j*tds.evals[:, i]/tds.dt
    for k in ks:
        print(i, k)
        sc = scattering(cyls, np.zeros(len(cyls)*(2*args.Nmode+1)), k=k, use_2cyl_modes=False, sum_index=args.Nmode)
        A = sc.make_scattering_blocks()
        if i == 0:
            condnum0 += [np.linalg.cond(A)]
        if i == 1:
            condnum1 += [np.linalg.cond(A)]
        if i == 2:
            condnum2 += [np.linalg.cond(A)]
    
runtime = time() - runtime
log.info('this run took %f seconds', runtime)
log.info('used 2*%d+1 modes for each cyl', tds.modes)
print('cond num computation done')
print('writing data')

#write out the data
fname = 'condnum.dat'
arr = [[np.abs(tds.ks[i, 0]), condnum0[i], np.abs(tds.ks[i, 1]), condnum1[i], np.abs(tds.ks[i, 2]), condnum2[i]] for i in range(len(ks))]
np.savetxt(os.path.join(dataout, fname), arr)

#copy the config file to the data output directory
os.system('cp '+args.cfg+' '+os.path.join(dataout, 'config_copy.cfg'))

#write the circle data to the data output directory
pickle.dump(cyls, open(os.path.join(dataout, "cyls.p"), "wb"))

print('done')
