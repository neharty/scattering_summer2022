import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import scattering, cyl

def uniform_grid_3x3(radius, spacing):
    # 3x3 grid of cylinders
    bcs=np.array([None for i in range(9)])
    for i in range(9):
        if i%3 == 0:
            bcs[i] = 'd' 
        elif i%3 == 1:
            bcs[i] = 'n'
        elif i%3 == 2:
            bcs[i] = 'i'
        else:
            None

    return uniform_grid_NxN(3, bcs, radius, spacing)

def uniform_grid_5x5(radius, spacing):
    # 5x5 grid of cylinders
    bcs = np.array([None for i in range(25)])
    for i in range(25):
        if i%5 == 0 or i%5 == 3:
            bcs[i] = 'd' 
        elif i%5 == 1 or i%5 == 4:
            bcs[i] = 'n'
        elif i%5 == 2:
            bcs[i] = 'i'
        else:
            None 

    return uniform_grid_NxN(5, bcs, radius, spacing)

def uniform_grid_NxN(N, bcs, radius, spacing, sum_num=None):
    cyls = [None for i in range(N*N)]
    space = spacing+2*radius
    
    if N%2 == 1:
        coeffs = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        coeffs = np.arange(-N/2, N/2 + 1) + 0.5*spacing

    origins = np.array([[coeffs[i], coeffs[j]] for i in range(N) for j in range(N)])
    
    for i in range(N*N):
        cyls[i] = cyl(bcs[i%N], origins[i], radius, sum_num)

    return cyls

def staggered_grid_N():
    return None

def uniform_pentagon(penta_radius, cyl_radius, ang_shift):
    origins_cmplx = 1j*np.zeros(5)
    for i in range(5):
        origins_cmplx[i] = penta_radius*np.exp(1j*(2*np.pi*i/5 + ang_shift))
        
    origins = np.zeros((5, 2))
    
    origins[:, 0] = np.real(origins_cmplx)[:]
    origins[:, 1] = np.imag(origins_cmplx)[:]
    
    print(origins)
    bcs = ['d', 'n', 'i', 'd', 'n']
    radius = 0.3
    return scattering([cyl(i, bcs[i], pos=origins[i], radius = cyl_radius) for i in range(5)])

def uniform_grating(N, bcs, radius, spacing):
    cyls = [None for i in range(N)]
    space = spacing+2*radius
    
    if N%2 == 1:
        coeffs = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        coeffs = np.arange(-N/2, N/2 + 1) + 0.5*spacing

    origins = np.array([[0, coeffs[j]] for j in range(N)])
    
    for i in range(N):
        cyls[i] = cyl(bcs[i], origins[i], radius)

    return cyls    


