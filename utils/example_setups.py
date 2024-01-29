import numpy as np
#import matplotlib.pyplot as plt
import os
#from Ncyl.Ncyl_gmres import scattering, cyl
#from Ncyl.tds import scattering, cyl
from Ncyl.cqm import cyl

def make_bcs(pos, bc):
    # uniform bcs
    # permuted bcs
    # all random
    pass

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

def uniform_grid_NxN(N, bcs, radius, spacing, sum_num=None, impedance=None, center = None):
    cyls = [None for i in range(N*N)]
    if N%2 == 1:
        coeffs = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        coeffs = np.arange(-N/2, N/2 + 1) + 0.5
    
    if center==None:
        origins = np.array([[coeffs[i], coeffs[j]] for i in range(N) for j in range(N)])*spacing
    else:
        center = np.array(center)
        origins = np.array([[coeffs[i] + center[0], coeffs[j] + center[1]] for i in range(N) for j in range(N)])*spacing

    for i in range(N*N):
        if type(bcs) is list:
            cyls[i] = cyl(bcs[i%N], origins[i], radius, sum_index = sum_num, impedance = impedance)
        else: 
            cyls[i] = cyl(bcs, origins[i], radius, sum_index = sum_num, impedance = impedance)
            
    return cyls

def uniform_grid_random_bc(N, bcs, radius, spacing, sum_num=None, impedance=None, center=None):
    cyls = uniform_grid_NxN(N, ['d' for n in range(N)], radius, spacing, sum_num = sum_num, center=center) #placeholder to change bcs
    bcs = np.random.choice(bcs, size=len(cyls))
    for c in range(len(cyls)):
        cyls[c].bc = bcs[c]
        print(cyls[c].bc)
        if bcs[c] == 'i':
            cyls[c].impedance = impedance
    return cyls

def big_circle(N, bc, big_radius, small_radius, sum_num = None, impedance=None, center=None):
    angs = np.linspace(0, 2*np.pi, N, endpoint=False)
    if center is None:
        c = [0, 0]
    else:
        c = center
    points = [[c[0] + big_radius*np.cos(a), c[1] + big_radius*np.sin(a)] for a in angs]
    return [cyl(bc, points[i], small_radius, sum_index=sum_num, impedance=impedance) for i in range(N)]

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
    return [cyl(i, bcs[i], pos=origins[i], radius = cyl_radius) for i in range(5)]

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

def rectangular_lattice(Nx, Ny, bcs, radius, xspace, yspace, sum_num=None, impedance = None, center = None):
    #print('rect_lat imp', impedance)
    cyls = [None for i in range(Nx*Ny)]
    
    if Nx%2 == 1:
        xcoeffs = np.arange(-(Nx-1)/2, (Nx-1)/2 + 1)
    else:
        xcoeffs = np.arange(-Nx/2, Nx/2 + 1) + 0.5

    if Ny%2 == 1:
        ycoeffs = np.arange(-(Ny-1)/2, (Ny-1)/2 + 1)
    else:
        ycoeffs = np.arange(-Ny/2, Ny/2 + 1) + 0.5
    
    if center==None:
        origins = np.array([[xcoeffs[i]*xspace, ycoeffs[j]*yspace] for i in range(Nx) for j in range(Ny)])
    else:
        origins = np.array([[xcoeffs[i]*xspace+center[0], ycoeffs[j]*yspace+center[1]] for i in range(Nx) for j in range(Ny)])
    
    
    if type(bcs) is list:
        if len(bcs) == int(Nx):
            for i in range(Nx*Ny):
                cyls[i] = cyl(bcs[i%Nx], origins[i], radius, sum_index = sum_num, impedance = impedance)
        else:
            for i in range(Nx*Ny):
                cyls[i] = cyl(bcs[i], origins[i], radius, sum_index = sum_num, impedance = impedance)
    else:
        cyls[i] = cyl(bcs, origins[i], radius, sum_index = sum_num, impedance = impedance)

    return cyls

def rectangular_lattice_1bc(Nx, Ny, bc, radius, xspace, yspace, sum_num=None, impedance = None, center = None):
    print('1bc imp', impedance)
    bcs = [bc for n in range(Nx)]
    return rectangular_lattice(Nx, Ny, bcs, radius, xspace, yspace, sum_num=sum_num, impedance=impedance, center=center)

def triangular_lattice(Nx, Ny, bcs, radius, xspace, yspace, sum_num=None, impedance = None, center=None):
    #print(bcs)
    if type(bcs) is str:
        bcs=[bcs for i in range(Nx)]
    cyls_main = rectangular_lattice(Nx, Ny, bcs, radius, xspace, yspace, sum_num=sum_num, impedance = impedance, center=center)
    del bcs[-1]
    cyls_off = rectangular_lattice(Nx-1, Ny-1, bcs, radius, xspace, yspace, sum_num=sum_num, impedance = impedance, center=center)
    return cyls_main + cyls_off

def random_pos(Lxmin, Lxmax, Lymin, Lymax, N, bc, radius, dist, sum_num=None, impedance=None):
    #fully random position placer in a box
    allowed_domain_x = (Lxmin+radius, Lxmax-radius)
    allowed_domain_y = (Lymin+radius, Lymax-radius)

    xpos = np.zeros(N)
    ypos = np.zeros(N)

    xpos[0], ypos[0] = np.random.uniform(*allowed_domain_x), np.random.uniform(*allowed_domain_y)
    for i in range(1, N):
        #generate a new point
        xtest, ytest = np.random.uniform(*allowed_domain_x), np.random.uniform(*allowed_domain_y)
        ztest = xtest + 1j*ytest
        #print(i)
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

def random_hexagon(Nx, Ny, bc, radius, dist, minspace=0, sum_num=None, impedance=None, center=None):
    ctemp = triangular_lattice(Nx, Ny, ['d' for n in range(Nx)], radius, dist, dist, sum_num=None, impedance=None, center=center)

    cnum = len(ctemp)
    pos = np.array([c.pos for c in ctemp])
    xpos = pos[:, 0]
    ypos = pos[:, 1]
    
    paverage = np.inf
    pvariance = np.inf
    
    def perturb():
        for i in range(cnum):            
            #generate a perturbation of the ith point
            r = np.sqrt(np.random.uniform(0, (dist/np.sqrt(2) - 2*radius-minspace)**2))
            theta = np.random.uniform(0, 2*np.pi)
            ztest = xpos[i] + 1j*ypos[i] + r*np.exp(1j*theta)
    
            #test new point against all other points
            idx = np.delete(np.arange(cnum), i)
            z = xpos[idx] + 1j*ypos[idx]
            allpoints = np.all(np.abs(z - ztest) > (2*radius + minspace))
    
            while not allpoints:
                #if the new point doesnt work, retry until it does
                r = np.sqrt(np.random.uniform(0, (dist/np.sqrt(2) - 2*radius-minspace))**2)
                theta = np.random.uniform(0, 2*np.pi)
                ztest = xpos[i] + 1j*ypos[i] + r*np.exp(1j*theta)
                allpoints = np.all(np.abs(z - ztest) > (2*radius + minspace))
        
            
            #set the ith point
            xpos[i], ypos[i] = ztest.real, ztest.imag
    
    #do while loop
    perturb()
    caverage = np.sqrt(np.mean(xpos)**2 + np.mean(ypos)**2)
    cvariance = np.var([[xpos[i], ypos[i]] for i in range(cnum)])
    #print(caverage, cvariance)
    while np.abs(paverage - caverage) >= 1e-3 and np.abs(pvariance - cvariance) >= 1e-3:
        paverage, pvariance = caverage, cvariance
        perturb()
        caverage = np.sqrt(np.mean(xpos)**2 + np.mean(ypos)**2)
        cvariance = np.var([[xpos[i], ypos[i]] for i in range(cnum)])
        #print(caverage, cvariance)
    
    origins = np.array([[xpos[i], ypos[i]] for i in range(cnum)])
    
    cyls = [None for i in range(cnum)]
    for i in range(cnum):
        cyls[i] = cyl(bc, origins[i], radius, sum_index=sum_num, impedance = impedance)
    
    return cyls

def random_hexagon_square(N, bc, radius, dist, minspace = 0, sum_num=None, impedance=None, center=None):
    return random_hexagon(N, N, bc, radius, dist, minspace, sum_num, impedance, center)

def all_random_hexagon(N, bcs, radius, dist, sum_num = None, impedance = None, center=None):
    cyls = random_hexagon_square(N, 'd', radius, dist, sum_num = sum_num, center=center) #placeholder to change bcs
    bcs = np.random.choice(bcs, size=len(cyls))
    for c in range(len(cyls)):
        cyls[c].bc = bcs[c]
        if bcs[c] == 'i':
            cyls[c].impedance = impedance
    return cyls

def vol_frac_allrandom_box(Vfrac, Lxmin, Lxmax, Lymin, Lymax, bc, radius, dist, maxtries=10, sum_num = None, impedance = None):
    Vbox = (Lxmax-Lxmin)*(Lymax - Lymin)
    N = int(np.floor(Vfrac*Vbox/(radius**2*np.pi)))
    print(N, 'volume fraction is ' + str(N*radius**2*np.pi/Vbox))
    return random_pos(Lxmin, Lxmax, Lymin, Lymax, N, bc, radius, dist, sum_num = None, impedance = None)

def vol_frac_hexagon():
    pass

def random_circ(rmax, N, bc, radius, sum_num=None, impedance=None, center=None):
    #fully random position placer in a circle
    allowed_r = rmax-radius
    
    dist = 2*radius
    
    xpos = np.zeros(N)
    ypos = np.zeros(N)
    
    def pos_fn():
        r = np.sqrt(np.random.uniform(0, allowed_r**2))
        ang = np.random.uniform(0, np.pi*2)
        return r*np.cos(ang), r*np.sin(ang)
    
    xpos[0], ypos[0] = pos_fn()
    for i in range(1, N):
        #generate a new point
        xtest, ytest = pos_fn()
        ztest = xtest + 1j*ytest
        #print(i)
        for j in range(i):
            #test new point against all previous points
            z = xpos[:i] + 1j*ypos[:i]
            allpoints = np.all(np.abs(z - ztest) > dist)
            while not allpoints:
                #if the new point doesnt work, retry until it does
                xtest, ytest = pos_fn()
                ztest = xtest + 1j*ytest
                allpoints = np.all(np.abs(z - ztest) > dist)
        xpos[i], ypos[i] = xtest, ytest

    origins = np.array([[xpos[i], ypos[i]] for i in range(N)])

    cyls = [None for i in range(N)]
    for i in range(N):
        cyls[i] = cyl(bc, origins[i], radius, sum_index=sum_num, impedance = impedance)

    return cyls

def random_circ_hex(rmax, bc, radius, initspace, minspace=0, sum_num=None, impedance=None, center=None, measurement_points=None, remove_center=False):
    spc = initspace
    N = int((2*rmax - 2*radius)/spc)
    print(N)
    ctmp = random_hexagon_square(N, bc, radius, spc, minspace, sum_num, impedance, center)
    
    #delete circles overlapping a measurement point
    if measurement_points is not None:
        X, Y = measurement_points
        idxs = []
        for i in range(len(ctmp)):
            c = ctmp[i]
            p = c.pos
            rad = c.radius
            dist = np.sqrt((X - p[0])**2 + (Y - p[1])**2) 
            mask = dist <= rad
            if np.any(mask):
                idxs += [i]
                
        idxs = np.array(idxs)
        for i in idxs[::-1]:
            ctmp.remove(ctmp[i])
    
    rmask = np.array([np.linalg.norm(c.pos) < (rmax-radius) for c in ctmp])
    
    if remove_center:
        rmask2 = np.array([np.linalg.norm(c.pos) > radius for c in ctmp])
        cctmp = np.asarray(ctmp)[rmask*rmask2]
    else:
        cctmp=np.asarray(ctmp)[rmask]
        
    return cctmp.tolist()

def random_circ_hex_vol_frac(R, bc, r, rho, minspace=0, sum_num=None, impedance=None, center=None, measurement_points=None, remove_center=False):
    #actual number of circles needed
    print(round(rho*R**2/r**2))
    n_act = int(round(rho*R**2/r**2))
    
    ncurr = 0
    Nprev = 0.5*(1 + np.sqrt(8*R**2*rho/(np.pi*r**2) - 1))
    while ncurr < n_act:
        #overfill the circle with higher rho
        rho = rho*1.05
        
        N = max(0.5*(1 + np.sqrt(8*R**2*rho/(np.pi*r**2) - 1)), Nprev+1)
        Nprev = N
        spc = (2*R - 2*r)/(N-1)
        tmp = random_circ_hex(R, bc, r, spc, minspace, sum_num, impedance, center, measurement_points, remove_center)
        ncurr = len(tmp)
    
    if ncurr > n_act:
        #delete the unneeded circles
        darr = np.arange(0, len(tmp))
        np.random.shuffle(darr)
        darr = darr[:ncurr - n_act]
        darr.sort()
        darr = darr[::-1]
        for i in darr:
            del tmp[i]

    return tmp

def random_box_hex_vol_frac(Lx, Ly, bc, r, rho, minspace=0, sum_num=0, impedance=None, center=None, measurement_points=None, remove_center=False):
    pass

def random_circ_hex_vol_frac_legacy(R, bc, r, rho, sum_num=None, impedance=None, center=None, measurement_points=None):
    #actual number of circles needed
    #produces the incorrect number of circles, used for posterity
    n_act = int(rho*R**2/r**2) + 1
    
    ncurr = 0
    Nprev = 0.5*(1 + np.sqrt(8*R**2*rho/(np.pi*r**2) - 1))
    while ncurr < n_act:
        #overfill the circle with higher rho
        rho = rho*1.05
        
        N = max(0.5*(1 + np.sqrt(8*R**2*rho/(np.pi*r**2) - 1)), Nprev+1)
        Nprev = N
        spc = (2*R - 2*r)/(N-1)
        tmp = random_circ_hex(R, bc, r, spc, sum_num, impedance, center, measurement_points)
        ncurr = len(tmp)
    
    if ncurr > n_act:
        #delete the unneeded circles
        darr = np.arange(0, len(tmp))
        np.random.shuffle(darr)
        darr = darr[:ncurr - n_act]
        darr.sort()
        darr = darr[::-1]
        for i in darr:
            del tmp[i]

    return tmp

def fig17_pos(bc, radius, sum_num=None, impedance=None, clusterdir=True):
    #pos = np.loadtxt(os.environ.get('PYTHONPATH')+'utils/fig17_points.csv', delimiter=',')
    if clusterdir:
        pos = np.loadtxt('/home/n/nh287/scattering/utils/fig17_points.csv', delimiter=',')
    else:
        pos = np.loadtxt('/home/nick/scattering/utils/fig17_points.csv', delimiter=',')
    cyls = [None for i in range(len(pos))]
    for i in range(len(pos)):
        cyls[i] = cyl(bc, pos[i], radius, sum_index=sum_num, impedance=impedance)

    return cyls


