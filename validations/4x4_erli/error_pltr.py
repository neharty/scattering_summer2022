import numpy as np
import matplotlib.pyplot as plt
import os

#initialize
def get_point(idx):
    ptdict = {
        2 : (6,0),
        3 : (-3, 3),
        4 : (-3, 0),
        5 : (-3, -3),
        6 : (0, 6),
        7 : (0, 0),
        8 : (0, -6),
        9 : (3, 3), 
        10 : (3, 0),
        11 : (3, -3), 
        12 : (6, 0), 
        13 : (9, 0), 
        14 : (9, 0.314),
        15 : (9, 0.628), 
        16 : (9, 0.942), 
        17 : (9, 1.257), 
        18 : (9, 1.571), 
        19 : (9, 1.885), 
        20 : (9, 2.199), 
        21 : (9, 2.513),
        22 : (9, 2.827),
        23 : (9, 3.141), 
        24 : (10.5, 0),
        25 : (10.5, 0.314),
        26 : (10.5, 0.628), 
        27 : (10.5, 0.942), 
        28 : (10.5, 1.257), 
        29 : (10.5, 1.571), 
        30 : (10.5, 1.885), 
        31 : (10.5, 2.199), 
        32 : (10.5, 2.513),
        33 : (10.5, 2.827),
        34 : (10.5, 3.141), 
        35 : (12, 0),
        36 : (12, 0.314),
        37 : (12, 0.628), 
        38 : (12, 0.942), 
        39 : (12, 1.257), 
        40 : (12, 1.571), 
        41 : (12, 1.885), 
        42 : (12, 2.199), 
        43 : (12, 2.513),
        44 : (12, 2.827), 
        45 : (12, 3.141), 
        }
    return ptdict[idx]

def key(idx):
    x, y = get_point(idx)
    return f'{x:.3f}_{y:.3f}.dat' 

bckey = {
    'n':'neuneu',
    'd':'dirdir', 
    'dndn': 'dirneu',
    'ndnd': 'neudir'}

l = 2 
inc = 3
bc='dndn'

 #reference soln

#%%
# error vs modes
for inc in [3, 4]:
    tint = 'irk5'
    srctype = 'planewave'
    if inc == 3:
        incang = -np.pi
        label= r'$\theta_{inc}=-\pi$'
    if inc == 4:
        incang = -np.pi-np.pi/3
        label= r'$\theta_{inc}=-4\pi/3$'
    
    refpath = '/home/nick/scattering/validations/4x4_erli/generic/'
    refpath = os.path.join(refpath, 
                           bckey[bc],
                           'lrep_'+str(l), 
                           'inc_'+str(inc)
                           )
    keyidx = 25
    fname = key(keyidx)
    refarr = np.loadtxt(os.path.join(refpath, fname))
    
    err = []
    Narr = [2, 4, 8, 10, 11, 12, 13, 14, 15, 16]
    for N in Narr:
        dpath = '/home/nick/scattering/validations/4x4_erli/data/gmres'
        dpath = os.path.join(dpath, 
                             tint, 
                             'srctype_'+srctype,
                             'bcs_'+bc,
                             'incang_'+str(incang),
                             'Nmode_'+str(N), 
                             'Ntime_14')
        fname = key(keyidx)
        arr = np.loadtxt(os.path.join(dpath, fname))
        err += [np.max(np.abs(arr[:, 1] - refarr[:, 1]))]
        
    plt.semilogy(Narr, err, '-', label=label)
    plt.xlabel('M')
    plt.ylabel('error')
    plt.legend()
plt.title('Convergence in M, using IRK5, (x, y) = '+str(get_point(keyidx)))
plt.savefig('/home/nick/scattering/proposal_imgs/s4/modeconverge.pdf')
plt.show()
        
#%%
# error vs timestep
tint = 'irk5'
srctype = 'planewave'
if inc == 3:
    incang = -np.pi
if inc == 4:
    incang = -np.pi-np.pi/3

err = []
Narr = np.arange(5, 15)
maxpow=14
end = 1
refarr = refarr[:int(end*len(refarr))]
for N in Narr:
    dpath = '/home/nick/scattering/validations/4x4_erli/data/'
    dpath = os.path.join(dpath, 
                         tint, 
                         'srctype_'+srctype,
                         'bcs_'+bc,
                         'incang_'+str(incang),
                         'Nmode_20', 
                         'Ntime_'+str(N))
    fname = key(keyidx)
    arr = np.loadtxt(os.path.join(dpath, fname))
    
    samepoints = refarr[::int(2**(maxpow - N)), 1]
    err += [np.max(np.abs(arr[:, 1] - samepoints))]

plt.loglog(40/2**Narr, 1*(40/2**Narr)**5, 'k--')
plt.loglog(40/2**Narr, err, '.-', label=tint)
plt.xlabel(r'$\Delta t$')
plt.ylabel('error')
plt.legend()
plt.show()