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

uvalr = np.zeros((*u_mudiff.shape, 21))
uvali = np.zeros((*u_mudiff.shape, 21))
uout = np.zeros((7, 21))
err = np.zeros((*u_mudiff.shape, 21))
for root, dirs, files in os.walk(rootdir):
    for d in dirs:
        #load my data
        try:
            u = np.loadtxt(rootdir+'mydat/'+d+'/U_r.dat') + 1j*np.loadtxt(rootdir+'mydat/'+d+'/U_i.dat')
            utmp = np.diag(np.abs(u))
            print(utmp.shape)
            d = int(d)
            uout[:, d] = utmp 
            #compute the error

            uvalr[:, :, d] = u.real
            uvali[:, :, d] = u.imag
            err[:, :, d] = np.abs(u - u_mudiff)/np.abs(u_mudiff)
        except Exception as e:
            print(e)
pos = np.loadtxt('/home/nick/scattering/tests/mu-diff_tests/qual/mpos.dat')

for i in range(7):
    plt.semilogy(np.arange(21), err[i,i,:], '.-', label=str((pos[0, i], pos[1, i])))
plt.xlabel(r'$N_l$')
plt.ylabel(r'$|u_{sc} - u_\mu|/|u_\mu|$')
plt.legend()
plt.tight_layout()
plt.show()
plt.clf()

'''
fig, ax = plt.subplots(2,1)
for i in range(7):
    #ax[0].plot(np.arange(15), uvalr[i,i,:15], '.-', label=str((pos[0, i], pos[1, i])))
    #ax[1].plot(np.arange(15), uvali[i,i,:15], '.-', label=str((pos[0, i], pos[1, i])))
    ax[0].semilogy(np.arange(15), np.abs(uvalr[i, i, :15] + 1j*uvali[i, i, :15]), '.-', label=str((pos[0, i], pos[1, i])))

plt.xlabel(r'$N_l$')
#plt.ylabel(r'$|u_{sc}|$')
ax[0].set_ylabel(r'$Re(u_{sc})$')
ax[1].set_ylabel(r'$Im(u_{sc})$')
plt.legend(bbox_to_anchor=(1, 1.5))
plt.tight_layout()
plt.show()
plt.clf()
'''

outpath=rootdir

#rewrite data out
for i in range(7):
    arr = [[j, uout[i, j]] for j in range(21)]
    np.savetxt(
        outpath+'{:.3f}'.format(pos[0, i])+'_'+'{:.3f}'.format(pos[1, i])+'_usc.dat',
        arr
        )  
