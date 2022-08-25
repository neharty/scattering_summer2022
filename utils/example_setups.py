import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../Ncyl/')
from Ncyl_gmres import scattering, cyl

def uniform_grid_3x3(radius, spacing, freq, sum_tol = 1e-12, gmres_tol = 1e-8, sum_index = None):
    # 3x3 grid of cylinders
    cyls = [None for i in range(9)]
    space = spacing + 2*radius
    origins = np.array([[-space, -space], [-space, 0], [-space, space], 
            [0, -space], [0, 0],[0, space], 
            [space, -space], [space, 0], [space, space]])
    for i in range(9):
        origins[i] = np.array(origins[i])

    for i in range(9):
        bc = 'd'
        if i%3 == 0:
            bc = 'd' 
        elif i%3 == 1:
            bc = 'n'
        elif i%3 == 2:
            bc = 'i'
        else:
            None

        cyls[i] = cyl(bc, origins[i], radius)

    return scattering(cyls, freq=freq, sum_tol=sum_tol, gmres_tol=gmres_tol, sum_index = sum_index)

def uniform_grid_5x5(radius, spacing, freq, sum_tol = 1e-12, gmres_tol = 1e-8, sum_index = None):
    # 5x5 grid of cylinders
    cyls = [None for i in range(25)]
    space = spacing + 2*radius
    origins = np.array([[-2*space, -2*space], [-2*space, -space], [-2*space, 0], [-2*space, space], [-2*space, 2*space],
        [-space, -2*space], [-space, -space], [-space, 0], [-space, space], [-space, 2*space],
        [0, -2*space], [0, -space], [0, 0], [0, space], [0, 2*space], 
        [space, -2*space], [space, -space], [space, 0], [space, space], [space, 2*space], 
        [2*space, 2*space], [2*space, -space], [2*space, 0], [2*space, space], [2*space, 2*space]])

    for i in range(25):
        origins[i] = np.array(origins[i])

    for i in range(25):
        bc = 'd'
        if i%5 == 0 or i%5 == 3:
            bc = 'd' 
        elif i%5 == 1 or i%5 == 4:
            bc = 'n'
        elif i%5 == 2:
            bc = 'i'
        else:
            None

        cyls[i] = cyl(bc, origins[i], radius)

    return scattering(cyls, freq=freq, sum_tol=sum_tol, gmres_tol=gmres_tol, sum_index = sum_index)

def uniform_grid_NxN(N, bcs, radius, spacing, freq, sum_tol = 1e-12, gmres_tol = 1e-8, sum_index = None, method='gmres'):
    cyls = [None for i in range(N*N)]
    space = spacing+2*radius
    
    if N%2 == 1:
        coeffs = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        coeffs = np.arange(-N/2, N/2 + 1)

    origins = np.array([[coeffs[i], coeffs[j]] for i in range(N) for j in range(N)])
    
    for i in range(N*N):
        cyls[i] = cyl(bcs[i], origins[i], radius)

    return scattering(cyls, freq=freq, sum_tol=sum_tol, gmres_tol=gmres_tol, sum_index=sum_index, method=method)


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

def uniform_grating(N, bcs, radius, spacing, freq, sum_tol = 1e-12, gmres_tol = 1e-8, sum_index = None, method='gmres'):
    cyls = [None for i in range(N)]
    space = spacing+2*radius
    
    if N%2 == 1:
        coeffs = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        coeffs = np.arange(-N/2, N/2 + 1)

    origins = np.array([[0, coeffs[j]] for j in range(N)])
    
    for i in range(N):
        cyls[i] = cyl(bcs[i], origins[i], radius)

    return scattering(cyls, freq=freq, sum_tol=sum_tol, gmres_tol=gmres_tol, sum_index=sum_index, method=method) 
    


