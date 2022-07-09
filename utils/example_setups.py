import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../Ncyl')
from Ncyl import scattering, cyl

def uniform_grid_3x3(radius, spacing):
    # 3x3 grid of cylinders
    cyls = [None for i in range(9)]
    radius = 0.2
    space = 1.5
    origins = np.array([[-space, -space], [-space, 0], [-space, space], 
            [0, -space], [0, 0],[0, space], 
            [space, -space], [space, 0], [space, space]])
    for i in range(9):
        origins[i] = np.array(origins[i])
    print(origins)

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

        cyls[i] = cyl(i, bc, pos=origins[i], radius=radius)

    return scattering(cyls)

def uniform_grid_5x5(radius, spacing):
    # 5x5 grid of cylinders
    cyls = [None for i in range(9)]
    radius = 0.2
    space = 1.5
    origins = np.array([[-space, -space], [-space, 0], [-space, space], 
            [0, -space], [0, 0],[0, space], 
            [space, -space], [space, 0], [space, space]])
    for i in range(9):
        origins[i] = np.array(origins[i])
    print(origins)

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

        cyls[i] = cyl(i, bc, pos=origins[i], radius=radius)

    return scattering(cyls)



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

def uniform_grating():
    return None


