import numpy as np
import matplotlib.pyplot as plt
#import sys
#sys.path.append('/home/n/nh287/scattering_summer2022/Ncyl/')
from Ncyl.Ncyl_gmres import scattering, cyl

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

def uniform_grid_NxN(N, bcs, radius, spacing, sum_num=None, impedance=None):
    cyls = [None for i in range(N*N)]
    
    if N%2 == 1:
        coeffs = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        coeffs = np.arange(-N/2, N/2 + 1) + 0.5

    origins = np.array([[coeffs[i], coeffs[j]] for i in range(N) for j in range(N)])*spacing
    
    for i in range(N*N):
        cyls[i] = cyl(bcs[i%N], origins[i], radius, sum_index = sum_num, impedance = impedance)

    return cyls

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

def uniform_grating(N, bcs, radius, spacing, sum_num = None, impedance = None):
    cyls = [None for i in range(N)]
    space = spacing+2*radius
    
    if N%2 == 1:
        coeffs = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        coeffs = np.arange(-N/2, N/2 + 1) + 0.5

    origins = np.array([[coeffs[j], 0] for j in range(N)])*spacing
    
    for i in range(N):
        cyls[i] = cyl(bcs[i], origins[i], radius, sum_index=sum_num, impedance = impedance)

    return cyls    

def rectangular_lattice(Nx, Ny, bcs, radius, xspace, yspace, sum_num=None, impedance = None):
    cyls = [None for i in range(Nx*Ny)]
    
    if Nx%2 == 1:
        xcoeffs = np.arange(-(Nx-1)/2, (Nx-1)/2 + 1)
    else:
        xcoeffs = np.arange(-Nx/2, Nx/2 + 1) + 0.5

    if Ny%2 == 1:
        ycoeffs = np.arange(-(Ny-1)/2, (Ny-1)/2 + 1)
    else:
        ycoeffs = np.arange(-Ny/2, Ny/2 + 1) + 0.5

    origins = np.array([[xcoeffs[i]*xspace, ycoeffs[j]*yspace] for i in range(Nx) for j in range(Ny)])

    for i in range(Nx*Ny):
        cyls[i] = cyl(bcs[i%Nx], origins[i], radius, sum_index = sum_num, impedance = impedance)

    return cyls

def triangular_lattice(Nx, Ny, bcs, radius, xspace, yspace, sum_num=None, impedance = None):
    cyls_main = rectangular_lattice(Nx, Ny, bcs, radius, xspace, yspace, sum_num=sum_num, impedance = impedance)
    cyls_off = rectangular_lattice(Nx-1, Ny-1, bcs, radius, xspace, yspace, sum_num=sum_num, impedance = impedance)
    return cyls_main + cyls_off

def random_pos(Lxmin, Lxmax, Lymin, Lymax, cutoff, N, bc, radius, sum_num=None, impedance=None):
    
    allcyls = False
    while not allcyls:
        xpos = np.random.uniform(low=Lxmin+radius, high=Lxmax-radius, size=2*N)
        ypos = np.random.uniform(low=Lymin+radius, high=Lymax-radius, size=2*N)
        
        cylidx = np.zeros(N)

        dist_matrix = np.zeros((2*N, 2*N))
        for i in range(2*N):
            for j in range(2*N):
                dist_matrix[i, j] = np.sqrt((xpos[i] - xpos[j])**2 + (ypos[i] - ypos[j])**2)
        
        mask = dist_matrix >= cutoff

        for i in range(2*N):
            mask[i , i] = True
        
        cylctr = 0
        for i in range(2*N):
            if np.all(mask[i, :]):
                cylidx[cylctr] = i
                cylctr += 1
            if cylctr >= N:
                allcyls = True
                break
     
    origins = np.array([[xpos[idxs[i][0]], ypos[idxs[i][1]]] for i in range(N)])

    cyls = [None for i in range(N)]
    for i in range(N):
        cyls[i] = cyl(bc, origins[i], radius, sum_index=sum_num, impedance = impedance)

    return cyls

