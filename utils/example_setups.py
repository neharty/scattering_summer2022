import numpy as np
import matplotlib.pyplot as plt
import os
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

def random_pos(Lxmin, Lxmax, Lymin, Lymax, N, bc, radius, dist, sum_num=None, impedance=None):
    allowed_domain_x = (Lxmin+radius, Lxmax-radius)
    allowed_domain_y = (Lymin+radius, Lymax-radius)

    xpos = np.zeros(N)
    ypos = np.zeros(N)

    xpos[0], ypos[0] = np.random.uniform(*allowed_domain_x), np.random.uniform(*allowed_domain_y)
    for i in range(1, N):
        #generate a new point
        xtest, ytest = np.random.uniform(*allowed_domain_x), np.random.uniform(*allowed_domain_y)
        ztest = xtest + 1j*ytest
        for j in range(i):
            #test new point against all previous points
            z = xpos[:i] + 1j*ypos[:i]
            allpoints = np.all(np.abs(z - ztest) > dist)
            while not allpoints:
                #if the new point doesnt work, retry until it does
                xtest, ytest = np.random.uniform(*allowed_domain_x), np.random.uniform(*allowed_domain_y)
                ztest = xtest + 1j*ytest
                allpoints = np.all(np.abs(z - ztest) > dist)
        xpos[i], ypos[i] = xtest, ytest

    origins = np.array([[xpos[i], ypos[i]] for i in range(N)])

    cyls = [None for i in range(N)]
    for i in range(N):
        cyls[i] = cyl(bc, origins[i], radius, sum_index=sum_num, impedance = impedance)

    return cyls

def fig17_pos(bc, radius, sum_num=None, impedance=None):
    pos = np.loadtxt(os.environ.get('PYTHONPATH')+'/utils/fig17_points.csv', delimiter=',')
    cyls = [None for i in range(len(pos))]
    for i in range(len(pos)):
        cyls[i] = cyl(bc, pos[i], radius, sum_index=sum_num, impedance=impedance)

    return cyls


