import numpy as np
from Ncyl.cqm import time_domain_scattering
from integrators.integrators import IRK3, IRK5
import pickle
import argparse
import os
import ast

parser = argparse.ArgumentParser()
parser.add_argument('cyl_path')
parser.add_argument('wv_path')
parser.add_argument('arg_path')
parser.add_argument('points')
parser.add_argument('outp_path')

args = parser.parse_args() 

#get cyls and waveform setup from pickled objects
cyls = pickle.load(open(args.cyl_path, 'rb'))
wv = pickle.load(open(args.wv_path, 'rb'))

#args file contains the simulation args
#from https://stackoverflow.com/questions/68122543/saving-and-reading-kwargs-from-a-file
kwargs = {}

with open(args.arg_path, "r") as f:
    for line in f.readlines():
        k, v = line.replace('\n','').split('=')
        kwargs.update({k.strip(): ast.literal_eval(v.strip())})

print(kwargs)
T = kwargs['T']
del kwargs['T'] #should be only kwargs
print(kwargs)

tds = time_domain_scattering(cyls, wv, T, **kwargs)

#points should be a 2-column data file
pts = np.loadtxt(args.points)

x = pts[:, 0]
y = pts[:, 1]
X, Y = np.meshgrid(x, y)
U = tds.make_U(X, Y)