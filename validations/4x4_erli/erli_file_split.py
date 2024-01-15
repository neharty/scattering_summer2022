import numpy as np
import os

def key(idx):
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
    
    x, y = ptdict[idx]
    return f'{x:.3f}_{y:.3f}.dat' 

base = '/home/nick/scattering/validations/4x4_erli/generic/'
for bc in ['dirdir', 'dirneu', 'neudir', 'neuneu']:
    for l in [1, 2]:
        for i in [3, 4]:
            flname = base+bc+'_lrep'+str(l)+'_xx7_Inc'+str(i)+'_tt14.dat'
            outp = os.path.join(base,
                                bc,
                                'lrep_'+str(l),
                                'inc_'+str(i))
            os.makedirs(outp, exist_ok=True)
            tmpdat = np.loadtxt(flname)
            for p in range(1, len(tmpdat[0])):
                parsedat = [[tmpdat[i, 0], tmpdat[i, p]] for i in range(len(tmpdat[:, 0]))]
                fname = os.path.join(outp, key(p + 1))
                np.savetxt(fname, parsedat)
            