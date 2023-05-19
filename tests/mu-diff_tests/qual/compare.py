import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from Ncyl.tds import scattering

parser = argparse.ArgumentParser()
parser.add_argument('dir', help='root directory')
rootdir = parser.parse_args().dir
print(rootdir)
u_mudiff = np.loadtxt(rootdir+'Ur.csv', delimiter=',') + 1j*np.loadtxt(rootdir+'Ui.csv', delimiter=',')

err = np.zeros((*u_mudiff.shape, 21))
for root, dirs, files in os.walk(rootdir):
    for d in dirs:
        #load my data
        try:
            u = np.loadtxt(rootdir+'mydat/'+d+'/U_r.dat') + 1j*np.loadtxt(rootdir+'mydat/'+d+'/U_i.dat')
            #compute the error
            d = int(d)
            err[:, :, d] = np.abs(u - u_mudiff)/np.abs(u_mudiff)
        except:
            pass
pos = np.loadtxt('/home/nick/scattering/tests/mu-diff_tests/qual/mpos.dat')
for i in range(7):
    plt.semilogy(np.arange(21), err[i,i,:], '.-', label=str((pos[0, i], pos[1, i])))
plt.xlabel(r'$N_l$')
plt.ylabel(r'$|u_{sc} - u_\mu|/|u_\mu|$')
plt.legend()
plt.tight_layout()
plt.show()
