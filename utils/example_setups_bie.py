import numpy as np
#import matplotlib.pyplot as plt
import os
#from Ncyl.Ncyl_gmres import scattering, cyl
#from Ncyl.tds import scattering, cyl
#from Ncyl.cqm import cyl
#from scatterers import cyl, starfish, fat_starfish, bean, strip
from scatterers import scat_select
from collections import deque

def set_scats(pos, sc_scale, bc, bcmode, scatterer_type, scrmode, rotation=0., rotmode=None, sum_num=None, impedance=None, rad_scale=1):
    modes=['uniform', 'repeated', 'permuted', 'random']
    
    l = len(pos)
    #print(l)
    
    if type(bc) is str:
        bc = [bc]
        
    #set the boundary conditions
    if bcmode=='uniform':
        bcs = [bc[0] for i in range(l)]
    elif bcmode=='repeated':
        bcs = np.empty(l, dtype=str)
        n = len(bc)
        for i in range(l):
            bcs[i] = bc[i%n]
        bcs = [bcs[i].item() for i in range(l)]
    elif bcmode=='permuted':
        d = deque(bc)
        for i in range(l//len(bc)):
            d.rotate()
            bc += list(d)
        # append the remaining permutations
        d.rotate()
        bc += list(d)[:l%len(bc)]
        bcs = bc
    elif bcmode=='random':
        bcs = np.random.choice(bc, size=l)
        bcs = [bcs[i].item() for i in range(l)]
    
    if type(rotation) is float:
        rotation = [rotation]
    
    #set the rotations   
    if rotmode=='uniform':
        rots = [rotation[0] for i in range(l)]
    elif rotmode=='repeated':
        rots = np.empty(l, dtype=float)
        n = len(rotation)
        for i in range(l):
            rots[i] = rotation[i%n]
        rots = [rots[i].item() for i in range(l)]
    elif rotmode=='permuted':
        d = deque(rotation)
        for i in range(l//len(rotation)):
            d.rotate()
            rotation += list(d)
        # append the remaining permutations
        d.rotate()
        rotation += list(d)[:l%len(bc)]
        rots = rotation
    elif rotmode=='random':
        rots = np.random.uniform(0, 2, l)*np.pi
    elif rotmode=='array':
        rots = rotation
    elif rotmode == None:
        rots = np.zeros(l)
    
    #set scatterers
    if type(scatterer_type) is not str:
        scrsname = scatterer_type
    
    if scrmode == 'uniform':
        scrs = [scat_select(scatterer_type, bcs[i], pos[i], sc_scale, rots[i], sum_num, impedance, rad_scale) for i in range(l)]
    elif scrmode=='random':
        scrsname = np.random.choice(scatterer_type, size=l)
        scrsname = [scrsname[i].item() for i in range(l)]
        scrs = []
        for i in range(l):
            tmp = [scat_select(scrsname[i], bcs[i], pos[i], sc_scale, rots[i], sum_num, impedance, rad_scale)]
            scrs += tmp
    elif scrmode=='array':
        scrs = []
        for i in range(l):
            tmp = [scat_select(scrsname[i], bcs[i], pos[i], sc_scale, rots[i], sum_num, impedance, rad_scale)]
            scrs += tmp
        
    return scrs

def uniform_grid_NxN(N, radius, spacing, center=np.array([0, 0])):
    if N%2 == 1:
        coeffs = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        coeffs = np.arange(-N/2, N/2 + 1) + 0.5
    
    center = np.array(center)
    origins = np.array([[coeffs[i] + center[0], coeffs[j] + center[1]] for i in range(N) for j in range(N)])*spacing

    return origins

def big_circle(N, big_radius, small_radius, center=np.array([0,0])):
    angs = np.linspace(0, 2*np.pi, N, endpoint=False)
    c = center
    points = [[c[0] + big_radius*np.cos(a), c[1] + big_radius*np.sin(a)] for a in angs]
    return points

def uniform_pentagon(penta_radius, cyl_radius, ang_shift):
    # don't use this
    origins_cmplx = 1j*np.zeros(5)
    for i in range(5):
        origins_cmplx[i] = penta_radius*np.exp(1j*(2*np.pi*i/5 + ang_shift))
        
    origins = np.zeros((5, 2))
    
    origins[:, 0] = np.real(origins_cmplx)[:]
    origins[:, 1] = np.imag(origins_cmplx)[:]
    
    bcs = ['d', 'n', 'i', 'd', 'n']
    radius = 0.3
    return [cyl(i, bcs[i], pos=origins[i], radius = cyl_radius) for i in range(5)]

def uniform_grating(N, radius, spacing, center=np.array([0., 0.]), xory='x'):
    cyls = [None for i in range(N)]
    space = spacing+2*radius
    
    if N%2 == 1:
        coeffs = np.arange(-(N-1)/2, (N-1)/2 + 1)
    else:
        coeffs = np.arange(-N/2, N/2 + 1) + 0.5
    
    if xory == 'x':
        origins = np.array([[center[0]+coeffs[j], center[1]] for j in range(N)])*spacing
    elif xory == 'y':
        origins = np.array([[center[0], center[1]+coeffs[j]] for j in range(N)])*spacing
    
    return origins

def rectangular_lattice(Nx, Ny, radius, xspace, yspace, center=np.array([0., 0.])): 
    if Nx%2 == 1:
        xcoeffs = np.arange(-(Nx-1)/2, (Nx-1)/2 + 1)
    else:
        xcoeffs = np.arange(-Nx/2, Nx/2 + 1) + 0.5

    if Ny%2 == 1:
        ycoeffs = np.arange(-(Ny-1)/2, (Ny-1)/2 + 1)
    else:
        ycoeffs = np.arange(-Ny/2, Ny/2 + 1) + 0.5
    
    origins = np.array([[xcoeffs[i]*xspace+center[0], ycoeffs[j]*yspace+center[1]] for i in range(Nx) for j in range(Ny)])
    
    return origins

def triangular_lattice(Nx, Ny, radius, xspace, yspace, center = np.array([0., 0.])):
    cyls_main = list(rectangular_lattice(Nx, Ny, radius, xspace, yspace, center))
    cyls_off = list(rectangular_lattice(Nx-1, Ny-1, radius, xspace, yspace, center))
    return np.array(cyls_main + cyls_off)

def triangular_lattice_square(N, radius, space, center=np.array([0., 0.])):
    return triangular_lattice(N, N, radius, space, space, center)

def random_pos(Lxmin, Lxmax, Lymin, Lymax, N, radius, dist, center=np.array([0., 0.])):
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

    origins = np.array([[xpos[i] + center[0], ypos[i]+center[1]] for i in range(N)])
    
    return origins

def random_hexagon(Nx, Ny, radius, dist, minspace=0, center=np.array([0., 0.])):
    pos = triangular_lattice(Nx, Ny, radius, dist, dist) #dummy hexagonal configuration
    cnum = len(pos)
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
    
    origins = np.array([[xpos[i]+center[0], ypos[i]+center[1]] for i in range(cnum)])
    
    return origins

def random_hexagon_square(N, radius, dist, minspace = 0, center=np.array([0., 0.])):
    return random_hexagon(N, N, radius, dist, minspace, center)

def random_circ(rmax, N, bc, radius, center=np.array([0., 0.])):
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
        
    origins = np.array([[xpos[i]+center[0], ypos[i]+center[1]] for i in range(N)])
    return origins

def random_circ_hex(rmax, radius, initspace, minspace=0, center=np.array([0., 0.]), measurement_points=None, remove_center=False, rin=0):
    spc = initspace
    N = int((2*rmax - 2*radius)/spc)
    pos = random_hexagon_square(N, radius, spc, minspace, center).tolist()
    
    #delete circles overlapping a measurement point
    if measurement_points is not None:
        X, Y = measurement_points
        idxs = []
        for i in range(len(pos)):
            p = pos[i]
            rad = radius
            dist = np.sqrt((X - p[0])**2 + (Y - p[1])**2) 
            mask = dist <= rad
            if np.any(mask):
                idxs += [i]
                
        idxs = np.array(idxs)
        for i in idxs[::-1]:
            pos.remove(pos[i])
    
    rmask = np.array([np.linalg.norm(p) < (rmax-radius) for p in pos])
    if rin > 0:
        rinmask = np.array([np.linalg.norm(p) > (rin+radius) for p in pos])
    else:
        rinmask = np.array([True for p in pos])
    
    if remove_center:
        rmask2 = np.array([np.linalg.norm(p) > radius for p in pos])
        cctmp = np.asarray(pos)[rmask*rmask2*rinmask]
    else:
        cctmp=np.asarray(pos)[rmask*rinmask]
        
    return cctmp

def random_circ_hex_vol_frac(R, r, rho, minspace=0, center=np.array([0., 0.]), measurement_points=None, remove_center=False, rin=0):
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
        tmp = random_circ_hex(R, r, spc, minspace, center, measurement_points, remove_center, rin=rin).tolist()
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
    
    pos = np.array([[p[0]+center[0], p[1]+center[1]] for p in tmp])
    
    return pos

def random_box_hex(Lx, Ly, radius, initspace, minspace=0, center=np.array([0., 0.]), measurement_points=None, remove_center=False, Lxin=0, Lyin=0):
    spc = initspace
    Nx = int((2*Lx - 2*radius)/spc)
    Ny = int((2*Ly - 2*radius)/spc)
    ctmp = random_hexagon(Nx, Ny, radius, spc, minspace, center)
    
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
        
    return cctmp

def random_box_hex_vol_frac(Lx, Ly, r, rho, minspace=0, center=np.array([0., 0.]), measurement_points=None, remove_center=False, Lxin=0, Lyin=0):
    n_act = int(round(4*rho*(Lx*Ly-Lxin*Lyin)/(np.pi*r**2))) #circles needed for the specified volume fraction
    
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
        tmp = random_box_hex(Lx, Ly, r, spc, minspace, center, measurement_points, remove_center, Lxin=Lxin, Lyin=Lyin)
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
    
    pos = np.array([[c.pos[0] + center[0], c.pos[1]+center[1]] for c in tmp])
    return pos

def random_circ_hex_vol_frac_legacy(R, r, rho, center=np.array([0., 0.]), measurement_points=None):
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
        tmp = random_circ_hex(R, r, spc, center, measurement_points)
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


