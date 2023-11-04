import numpy as np
from scipy.special import hankel1, hankel2, h1vp, h2vp
from Ncyl.cqm import cyl

def ida_ishihara_goto(cyl, k, m):
    #TODO: need to fix
    a = cyl.radius
    #kstar = np.sqrt(3)*k
    #return 1j*h2vp(m, kstar*a)/hankel2(m, kstar*a)/np.sqrt(3)
    return 1/np.sqrt(3)

def nguyen(k, eps=1e-2):
    # nguyen et al 2015
    alpha = np.exp(-1j*np.pi/4)
    return -1/(eps*alpha*(k**0.5))

def M120_rapprox(k):
    #from dragna 2013
    #scaling is off
    A = np.array([1.4144, 1.0014, -3.336, 5.2545, 3.0317e1])*1e6
    lam = np.array([52.33, 494.61, 1702.5, 1832.7, 34000])
    
    tmp = 0j
    for i in range(len(A)):
        tmp += A[i]/(lam[i] + k)
    return 1/tmp

def M12L02_rapprox(k):
    #from dragna 2013
    #scaling is off
    A = np.array([-2.3877e5, 1.3051e6, 91175, -4.4016e6, 2.7118e7])
    lam = np.array([1.4464, 2.1303, 1263.8, 6531.1, 27199])
    
    tmp = 0j
    for i in range(len(A)):
        tmp += A[i]/(lam[i] + k)
    return 1/tmp

def M120(k):
    #from dragna 2013
    #exact impedance
    sig = 120
    mu = 0.459
    b = 0.632
    return -1/(1+mu*(sig/(-1j*k))**b)

def M12L02(k):
    #from dragna 2013
    #exact impedance
    sig = 12
    mu = 0.459
    b = 0.632
    nu = 0.673
    q = 0.618
    kl = k*(1+nu*(sig/(-1j*k))**q)
    tmp = np.tanh(-1j*kl*0.2) 
    return -tmp/(1+mu*(sig/(-1j*k))**b)    


