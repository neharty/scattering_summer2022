import numpy as np
#import matplotlib.pyplot as plt
import os
#from Ncyl.Ncyl_gmres import scattering, cyl
#from Ncyl.tds import scattering, cyl
from Ncyl.cqm import cyl
from collections import deque

def make_cyls(pos, bc, radius, mode, sum_num=None, impedance=None):
    modes=['uniform', 'repeated', 'permuted', 'random']
    
    l = len(pos)
    #print(l)
    
    if type(bc) is str:
        bc = [bc]
    
    if mode=='uniform':
        bcs = [bc[0] for i in range(l)]
    elif mode=='repeated':
        bcs = np.empty(l, dtype=str)
        n = len(bc)
        for i in range(l):
            bcs[i] = bc[i%n]
        bcs = [bcs[i].item() for i in range(l)]
    elif mode=='permuted':
        d = deque(bc)
        for i in range(l//len(bc)):
            d.rotate()
            bc += list(d)
        # append the remaining permutations
        d.rotate()
        bc += list(d)[:l%len(bc)]
        bcs = bc
    elif mode=='random':
        bcs = np.random.choice(bc, size=l)
        bcs = [bcs[i].item() for i in range(l)]
        
    cyls=[cyl(bcs[i], pos[i], radius, sum_num, impedance) for i in range(l)]
    return cyls

def uniform_grid_NxN(N, bcs, radius, spacing, sum_num=None, impedance=None, center = None, bc_mode='uniform'):
    if N%2 == 1:
        coeffs = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        coeffs = np.arange(-N/2, N/2 + 1) + 0.5
    
    if center==None:
        origins = np.array([[coeffs[i], coeffs[j]] for i in range(N) for j in range(N)])*spacing
    else:
        center = np.array(center)
        origins = np.array([[coeffs[i] + center[0], coeffs[j] + center[1]] for i in range(N) for j in range(N)])*spacing

    return make_cyls(origins, bcs, radius, bc_mode, sum_num, impedance)

def big_circle(N, bc, big_radius, small_radius, sum_num = None, impedance=None, center=None, bc_mode='uniform'):
    angs = np.linspace(0, 2*np.pi, N, endpoint=False)
    if center is None:
        c = [0, 0]
    else:
        c = center
    points = [[c[0] + big_radius*np.cos(a), c[1] + big_radius*np.sin(a)] for a in angs]

    return make_cyls(points, bc, bc_mode, sum_num, impedance)
    #return [cyl(bc, points[i], small_radius, sum_index=sum_num, impedance=impedance) for i in range(N)]

def uniform_pentagon(penta_radius, cyl_radius, ang_shift):
    origins_cmplx = 1j*np.zeros(5)
    for i in range(5):
        origins_cmplx[i] = penta_radius*np.exp(1j*(2*np.pi*i/5 + ang_shift))
        
    origins = np.zeros((5, 2))
    
    origins[:, 0] = np.real(origins_cmplx)[:]
    origins[:, 1] = np.imag(origins_cmplx)[:]
    
    bcs = ['d', 'n', 'i', 'd', 'n']
    radius = 0.3
    return [cyl(i, bcs[i], pos=origins[i], radius = cyl_radius) for i in range(5)]

def uniform_grating(N, bcs, radius, spacing, sum_num = None, impedance = None, bc_mode='uniform'):
    cyls = [None for i in range(N)]
    space = spacing+2*radius
    
    if N%2 == 1:
        coeffs = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        coeffs = np.arange(-N/2, N/2 + 1) + 0.5

    origins = np.array([[coeffs[j], 0] for j in range(N)])*spacing

    return make_cyls(origins, bcs, radius, bc_mode, sum_num, impedance)

def rectangular_lattice(Nx, Ny, bcs, radius, xspace, yspace, sum_num=None, impedance = None, center = None, bc_mode='uniform'): 
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
    
    return make_cyls(origins, bcs, radius, bc_mode, sum_num, impedance)

def triangular_lattice(Nx, Ny, bcs, radius, xspace, yspace, sum_num=None, impedance = None, center=None, bc_mode='uniform'):
    cyls_main = rectangular_lattice(Nx, Ny, bcs, radius, xspace, yspace, sum_num, impedance, center, bc_mode)
    cyls_off = rectangular_lattice(Nx-1, Ny-1, bcs, radius, xspace, yspace, sum_num, impedance, center, bc_mode)
    return cyls_main + cyls_off

def triangular_lattice_square(N, bcs, radius, space, sum_num=None, impedance = None, center=None, bc_mode='uniform'):
    return triangular_lattice(N, N, bcs, radius, space, space, sum_num, impedance, center, bc_mode='uniform')

def random_pos(Lxmin, Lxmax, Lymin, Lymax, N, bc, radius, dist, sum_num=None, impedance=None, bc_mode='uniform'):
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
    
    return make_cyls(origins, bc, radius, bc_mode, sum_num, impedance)

def random_hexagon(Nx, Ny, bc, radius, dist, minspace=0, sum_num=None, impedance=None, center=None, bc_mode='uniform'):
    ctemp = triangular_lattice(Nx, Ny, 'd', radius, dist, dist) #dummy hexagonal configuration
    cnum = len(ctemp)
    pos = np.array([c.pos for c in ctemp])
    xpos = pos[:, 0]
    ypos = pos[:, 1]
    
    paverage = np.inf
    pvariance = np.inf
    
    def perturb():
        for i in range(cnum):            
            #generate a perturbation of the ith point
            r = np.sqrt(np.random.uniform(0, (1.3*(dist/np.sqrt(2) - 2*radius-minspace))**2))
            theta = np.random.uniform(0, 2*np.pi)
            ztest = xpos[i] + 1j*ypos[i] + r*np.exp(1j*theta)
    
            #test new point against all other points
            idx = np.delete(np.arange(cnum), i)
            z = xpos[idx] + 1j*ypos[idx]
            allpoints = np.all(np.abs(z - ztest) > (2*radius + minspace))
    
            while not allpoints:
                #if the new point doesnt work, retry until it does
                r = np.sqrt(np.random.uniform(0, (1.3*(dist/np.sqrt(2) - 2*radius-minspace))**2))
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
    
    return make_cyls(origins, bc, radius, bc_mode, sum_num, impedance)

def random_hexagon_square(N, bc, radius, dist, minspace = 0, sum_num=None, impedance=None, center=None, bc_mode='uniform'):
    return random_hexagon(N, N, bc, radius, dist, minspace, sum_num, impedance, center, bc_mode)

def random_circ(rmax, N, bc, radius, sum_num=None, impedance=None, center=None, bc_mode='uniform'):
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
    
    return make_cyls(origins, bc, radius, bc_mode, sum_num, impedance)

def random_circ_hex(rmax, bc, radius, initspace, minspace=0, sum_num=None, impedance=None, center=None, measurement_points=None, remove_center=False, bc_mode='uniform', rin=0):
    spc = initspace
    N = int((2*rmax - 2*radius)/spc)
    ctmp = random_hexagon_square(N, bc, radius, spc, minspace, sum_num, impedance, center, bc_mode)
    
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
    if rin > 0:
        rinmask = np.array([np.linalg.norm(c.pos) > (rin+radius) for c in ctmp])
    else:
        rinmask = np.array([True for c in ctmp])
    
    if remove_center:
        rmask2 = np.array([np.linalg.norm(c.pos) > radius for c in ctmp])
        cctmp = np.asarray(ctmp)[rmask*rmask2*rinmask]
    else:
        cctmp=np.asarray(ctmp)[rmask*rinmask]
        
    return cctmp.tolist()

def random_circ_hex_vol_frac(R, bc, r, rho, minspace=0, sum_num=None, impedance=None, center=None, measurement_points=None, remove_center=False, bc_mode='uniform', rin=0):
    n_act = int(round(rho*(R**2 - rin**2)/r**2)) #circles needed for the specified volume fraction
    
    ncurr = 0
    Nprev = int(round(0.5*(1 + np.sqrt(8*(R**2-rin**2)*rho/(np.pi*r**2) - 1))))
    while ncurr < n_act:
        #overfill the circle with higher rho
        rho = rho*1.05    
        N = max(0.5*(1 + np.sqrt(8*(R**2-rin**2)*rho/(np.pi*r**2) - 1)), Nprev+1)
        Nprev = N
        spc = (2*R - 2*r)/(N-1)
        #dummy circles for position setting
        tmp = random_circ_hex(R, 'd', r, spc, minspace, sum_num, impedance, center, measurement_points, remove_center, rin=rin)
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
    
    pos = [[c.pos[0], c.pos[1]] for c in tmp]
    
    return make_cyls(pos, bc, r, bc_mode, sum_num, impedance)

def random_box_hex(Lx, Ly, bc, radius, initspace, minspace=0, sum_num=None, impedance=None, center=None, measurement_points=None, remove_center=False, bc_mode='uniform', Lxin=0, Lyin=0):
    spc = initspace
    Nx = int((2*Lx - 2*radius)/spc)
    Ny = int((2*Ly - 2*radius)/spc)
    ctmp = random_hexagon(Nx, Ny, bc, radius, spc, minspace, sum_num, impedance, center, bc_mode)
    
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
    
    xmask = np.array([np.abs(c.pos[0]) < (Lx-radius) for c in ctmp])
    if Lxin > 0:
        lxinmask = np.array([np.abs(c.pos[0]) > (Lxin+radius) for c in ctmp])
    else:
        lxinmask = np.array([True for c in ctmp])
        
    ymask = np.array([np.abs(c.pos[1]) < (Ly-radius) for c in ctmp])
    if Lyin > 0:
        lyinmask = np.array([np.abs(c.pos[1]) > (Lyin+radius) for c in ctmp])
    else:
        lyinmask = np.array([True for c in ctmp])
    
    boxmask = (xmask*ymask*lxinmask) | (xmask*ymask*lyinmask)
    
    if remove_center:
        rmask2 = np.array([np.linalg.norm(c.pos) > radius for c in ctmp])
        cctmp = np.asarray(ctmp)[boxmask*rmask2]
    else:
        cctmp=np.asarray(ctmp)[boxmask]
        
    return cctmp.tolist()

def random_box_hex_vol_frac(Lx, Ly, bc, r, rho, minspace=0, sum_num=0, impedance=None, center=None, measurement_points=None, remove_center=False, bc_mode='uniform', Lxin=0, Lyin=0):
    n_act = int(round(4*rho*(Lx-Lxin)*(Ly-Lyin)/(np.pi*r**2))) #circles needed for the specified volume fraction
    
    ncurr = 0
    gam = (Lx-Lxin)/(Ly-Lyin)
    Nxprev = int(round(0.25*(gam+1+np.sqrt(32*rho*(Lx-Lxin)**2/(np.pi*r**2)+(gam+1)**2-8*gam))))
    Nyprev = int(Ly*Nxprev/Lx)
    while ncurr < n_act:
        #overfill the circle with higher rho
        rho = rho*1.05    
        Nx = max(0.25*(gam+1+np.sqrt(32*rho*(Lx-Lxin)**2/(np.pi*r**2)+(gam+1)**2-8*gam)), Nxprev+1)
        Nxprev = Nx
        spc = (2*Lx - 2*r)/(Nxprev-1)
        #dummy circles for position setting
        tmp = random_box_hex(Lx, Ly, bc, r, spc, minspace, sum_num, impedance, center, measurement_points, remove_center, Lxin=Lxin, Lyin=Lyin)
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
    
    pos = [[c.pos[0], c.pos[1]] for c in tmp]
    
    return make_cyls(pos, bc, r, bc_mode, sum_num, impedance)

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
    #fig 17 from Antoine et al. 2008
    #pos = np.loadtxt(os.environ.get('PYTHONPATH')+'utils/fig17_points.csv', delimiter=',')
    if clusterdir:
        pos = np.loadtxt('/home/n/nh287/scattering/utils/fig17_points.csv', delimiter=',')
    else:
        pos = np.loadtxt('/home/nick/scattering/utils/fig17_points.csv', delimiter=',')
    cyls = [None for i in range(len(pos))]
    for i in range(len(pos)):
        cyls[i] = cyl(bc, pos[i], radius, sum_index=sum_num, impedance=impedance)

    return cyls


