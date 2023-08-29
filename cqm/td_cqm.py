from Ncyl.cqm import time_domain_scattering
from integrators.integrators import IRK3, IRK5
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('cyl_path')
parser.add_argument('wv_path')
parser.add_argument('arg_path')
parser.add_argument('points')
parser.add_argument('outp_path')

args = parser.parse_args() 

#get cyls and waveform setup from pickled objects
#cyls = pickle.load(args.cyl_path, 'rb')
#wv = pickle.load(args.wv_path, 'rb')

#args file contains the simulation args
#from https://stackoverflow.com/questions/68122543/saving-and-reading-kwargs-from-a-file
kwargs = {}

with open(args.arg_path, "r") as f:
    for line in f.readlines():
        k, v = line.replace('\n','').split('=')
        kwargs.update({k.strip(): v.strip()})

print(kwargs)
T = kwargs['T']
del kwargs['T'] #should be only kwargs
print(kwargs)

tds = time_domain_scattering(cyls, wv, T, **kwargs)


