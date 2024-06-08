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
parser.add_argument('Nmode', help='sum index', type=int)
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

#make data output folder
# dataout = os.path.join(
#         args.fout,
#         cylstr,
#         'volfrac_'+str(args.volfrac),
#         'Nmode_'+str(args.Nmode),
#         'seed_' + str(args.seed)
#         )

# if not os.path.exists(dataout):
#     os.makedirs(dataout)

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

tds = time_domain_scattering(
                        cyls, 
                        wv, 
                        T, 
                        sampnum = 2**int(config['TDS']['Nt']), 
                        timeint=tint,
                        modes = args.Nmode
                        )
tds.fzt_boundary_allcyls_highorder() #reduces the number of right hand sides

condnum0cqs = []
condnum1cqs = []
condnum2cqs = []
arrcqs = []

condnum0cqu = []
condnum1cqu = []
condnum2cqu = []
arrcqu = []

condnum0rks = []
condnum1rks = []
condnum2rks = []
arrrks = []

condnum0rku = []
condnum1rku = []
condnum2rku = []
arrrku = []

for i in range(3):
    ks = 1j*tds.evals[:, i]/tds.dt
    for k in ks:
        print(i, k)
        #cq, scaled
        sc = scattering(cyls, np.zeros(len(cyls)*(2*args.Nmode+1)), k=k, use_2cyl_modes=False, sum_index=args.Nmode)
        A = sc.make_scattering_blocks()
        if i == 0:
            if np.isfinite(np.abs(A)).all():
                condnum0cqs += [np.linalg.cond(A)]
            else:
                condnum0cqs += [0]
        if i == 1:
            if np.isfinite(np.abs(A)).all():
                condnum1cqs += [np.linalg.cond(A)]
            else:
                condnum1cqs += [0]
        if i == 2:
            if np.isfinite(np.abs(A)).all():
                condnum2cqs += [np.linalg.cond(A)]
            else:
                condnum2cqs += [0]
        
        #cq unscaled
        sc = scattering(cyls, np.zeros(len(cyls)*(2*args.Nmode+1)), k=k, use_2cyl_modes=False, sum_index=args.Nmode, use_reduced_system=False)
        A = sc.make_scattering_blocks()
        if i == 0:
            if np.isfinite(np.abs(A)).all():
                condnum0cqu += [np.linalg.cond(A)]
            else:
                condnum0cqu += [0]
        if i == 1:
            if np.isfinite(np.abs(A)).all():
                condnum1cqu += [np.linalg.cond(A)]
            else:
                condnum1cqu += [0]
        if i == 2:
            if np.isfinite(np.abs(A)).all():
                condnum2cqu += [np.linalg.cond(A)]
            else:
                condnum2cqu += [0]
        
        #real-k scaled
        sc = scattering(cyls, np.zeros(len(cyls)*(2*args.Nmode+1)), k=np.abs(k), use_2cyl_modes=False, sum_index=args.Nmode)
        A = sc.make_scattering_blocks()
        if i == 0:
            if np.isfinite(np.abs(A)).all():
                condnum0rks += [np.linalg.cond(A)]
            else:
                condnum0rks += [0]
        if i == 1:
            if np.isfinite(np.abs(A)).all():
                condnum1rks += [np.linalg.cond(A)]
            else:
                condnum1rks += [0]
        if i == 2:
            if np.isfinite(np.abs(A)).all():
                condnum2rks += [np.linalg.cond(A)]
            else:
                condnum2rks += [0]

        #real-k unscaled
        sc = scattering(cyls, np.zeros(len(cyls)*(2*args.Nmode+1)), k=np.abs(k), use_2cyl_modes=False, sum_index=args.Nmode, use_reduced_system=False)
        A = sc.make_scattering_blocks()
        if i == 0:
            if np.isfinite(np.abs(A)).all():
                condnum0rku += [np.linalg.cond(A)]
            else:
                condnum0rku += [0]
        if i == 1:
            if np.isfinite(np.abs(A)).all():
                condnum1rku += [np.linalg.cond(A)]
            else:
                condnum1rku += [0]
        if i == 2:
            if np.isfinite(np.abs(A)).all():
                condnum2rku += [np.linalg.cond(A)]
            else:
                condnum2rku += [0]

    
runtime = time() - runtime
log.info('this run took %f seconds', runtime)
log.info('used 2*%d+1 modes for each cyl', tds.modes)
print('cond num computation done')
print('writing data')

# save the data
fname = 'condnum.dat'

arrcqs = [[np.abs(ks[i, 0]), condnum0cqs[i], np.abs([i, 1]), condnum1cqs[i], np.abs(ks[i, 2]), condnum2cqs[i]] for i in range(len(ks))]
np.savetxt(os.path.join(dataout, 'cq_scaled', fname), arrcqs)

arrcqu = [[np.abs(ks[i, 0]), condnum0cqu[i], np.abs(ks[i, 1]), condnum1cqu[i], np.abs(ks[i, 2]), condnum2cqu[i]] for i in range(len(ks))]
np.savetxt(os.path.join(dataout, 'cq_unscaled', fname), arrcqu)

arrrks = [[np.abs(ks[i, 0]), condnum0rks[i], np.abs(ks[i, 1]), condnum1rks[i], np.abs(ks[i, 2]), condnum2rks[i]] for i in range(len(ks))]
np.savetxt(os.path.join(dataout, 'real_k_scaled', fname), arrrks)

arrrku = [[np.abs(ks[i, 0]), condnum0rku[i], np.abs(ks[i, 1]), condnum1rku[i], np.abs(ks[i, 2]), condnum2rku[i]] for i in range(len(ks))]
np.savetxt(os.path.join(dataout, 'real_k_unscaled', fname), arrrku)

# #copy the config file to the data output directory
# os.system('cp '+args.cfg+' '+os.path.join(dataout, 'config_copy.cfg'))

# #write the circle data to the data output directory
# pickle.dump(cyls, open(os.path.join(dataout, "cyls.p"), "wb"))

print('done')
