import numpy as np
from scipy.special import hankel1, hankel2, h1vp, h2vp
from Ncyl.cqm import cyl

def ida_ishihara_goto(cyl, k, m):
    a = cyl.radius
    kstar = np.sqrt(3)*k
    return 1j*h2vp(m, kstar*a)/hankel2(m, kstar*a)/np.sqrt(3)
    
    