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
parser.add_argument('Nmode_start', help='start sum index', type=int)
parser.add_argument('Nmode_stop', help='end sum index', type=int)
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

dataout = args.fout
os.makedirs(dataout, exist_ok=True)

#debug output file
log = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join('debug.log'), level=logging.DEBUG, filemode='w')
log.info(args)
# wave setup
print('generating wave')
wv = gaussian_norm_width(width=1, lag = 1) #dummy wave

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

print('num of cyls:', len(cyls))
log.info('number of cyls is %d', len(cyls))

#set up time domain sim
T = float(config['TDS']['T'])
tint = IRK5()
log.info('using IRK5')
print('setting up time domain simulation')

Ns = np.arange(args.Nmode_start, args.Nmode_stop+1)

tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = 2**int(config['TDS']['Nt']), 
                        timeint=tint,
                        modes = args.Nmode_stop
                        )
tds.set_timeshift()
tds.fzt_boundary_allcyls_highorder() # reduce the number of right hand sides
ks = 1j*tds.evals[:tds.rhs_maxidx, 1]/tds.dt # solved for k-values
fname = 'condnum.dat'

Narrcqs = []
Narrcqu = []
Narrrks = []
Narrrku = []

for N in Ns:
    condnum1cqs = []
    arrcqs = []

    condnum1cqu = []
    arrcqu = []

    condnum1rks = []
    arrrks = []

    condnum1rku = []
    arrrku = []
    
    for i in range(len(ks)):
        k = ks[i]
        print(N, k)
        #cq, scaled
        sc = scattering(cyls, np.zeros(len(cyls)*(2*N+1)), k=k, use_2cyl_modes=False, sum_index=N)
        A = sc.make_scattering_blocks()
        if np.isfinite(np.abs(A)).all():
            condnum1cqs += [np.linalg.cond(A)]
        else:
            condnum1cqs += [0]
        arrcqs += [[np.abs(ks[i]), condnum1cqs[i]]]
        np.savetxt(os.path.join(dataout, f'cq_scaled_{N}_'+fname), arrcqs)
        
        #cq unscaled
        sc = scattering(cyls, np.zeros(len(cyls)*(2*N+1)), k=k, use_2cyl_modes=False, sum_index=N, use_reduced_system=False)
        A = sc.make_scattering_blocks()
        if np.isfinite(np.abs(A)).all():
            condnum1cqu += [np.linalg.cond(A)]
        else:
            condnum1cqu += [0]
        arrcqu += [[np.abs(ks[i]), condnum1cqs[i]]]
        np.savetxt(os.path.join(dataout, f'cq_unscaled_{N}_'+fname), arrcqs)
        
        #real-k scaled
        sc = scattering(cyls, np.zeros(len(cyls)*(2*N+1)), k=np.abs(k), use_2cyl_modes=False, sum_index=N)
        A = sc.make_scattering_blocks()
        if np.isfinite(np.abs(A)).all():
            condnum1rks += [np.linalg.cond(A)]
        else:
            condnum1rks += [0]
        arrrks += [[np.abs(ks[i]), condnum1cqs[i]]]
        np.savetxt(os.path.join(dataout, f'real_k_scaled_{N}_'+fname), arrcqs)
    
        #real-k unscaled
        sc = scattering(cyls, np.zeros(len(cyls)*(2*N+1)), k=np.abs(k), use_2cyl_modes=False, sum_index=N, use_reduced_system=False)
        A = sc.make_scattering_blocks()
        if np.isfinite(np.abs(A)).all():
            condnum1rku += [np.linalg.cond(A)]
        else:
            condnum1rku += [0]
        arrrku += [[np.abs(ks[i]), condnum1cqs[i]]]
        np.savetxt(os.path.join(dataout, f'real_k_unscaled_{N}_'+fname), arrcqs)
        
    tmp = np.max(condnum1cqs)
    Narrcqs += [[N, tmp]]
    np.savetxt(os.path.join(dataout, 'cq_scaled_maxk.dat'), Narrcqs)
    
    tmp = np.max(condnum1cqu)
    Narrcqu += [[N, tmp]]
    np.savetxt(os.path.join(dataout, 'cq_unscaled_maxk.dat'), Narrcqu)
    
    tmp = np.max(condnum1rks)
    Narrrks += [[N, tmp]]
    np.savetxt(os.path.join(dataout, 'real_k_scaled_maxk.dat'), Narrrks)
    
    tmp = np.max(condnum1rku)
    Narrrku += [[N, tmp]]
    np.savetxt(os.path.join(dataout, 'real_k_unscaled_maxk.dat'), Narrrku)
    
runtime = time() - runtime
log.info('this run took %f seconds', runtime)
log.info('used 2*%d+1 modes for each cyl', tds.modes)
print('cond num computation done')

print('done')
