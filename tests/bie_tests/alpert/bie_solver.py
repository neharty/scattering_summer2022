# try to solve using alpert quadrature
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
#from Ncyl.cqm import scattering

# quadrature parameters
# for h^6|log h|
chi=[4.004884194926570e-03, 
      7.745655373336686e-02,
      3.972849993523248e-01,
      1.075673352915104e+00,
      2.003796927111872e+00, 
      -4.004884194926570e-03, 
      -7.745655373336686e-02,
      -3.972849993523248e-01,
      -1.075673352915104e+00,
      -2.003796927111872e+00]
    
w=[1.671879691147102e-02,
    1.636958371447360e-01,
    4.981856569770637e-01,
    8.372266245578912e-01,
    9.841730844088381e-01,
    1.671879691147102e-02,
    1.636958371447360e-01,
    4.981856569770637e-01,
    8.372266245578912e-01,
    9.841730844088381e-01]

m = 5
a = 3
l = 6
M = l + 3

#---------------------
# 2nd order alpert
# chi = [1.591549430918953e-01, -1.591549430918953e-01]
# w = [5.000000000000000e-01, 5.000000000000000e-01]

# m=1
# a=1
# l=2
# M=l+3
#------------------

# wavenumber
k = 2

def ll(i, j, N):
    # from Erli's code
    a = np.mod(i-j, N)
    b = np.mod(j-i, N)
    if a<b:
        return -a
    else:
        return b

def o_np(n, p):
    #use 1-indexing
    p-=1
    return np.floor(chi[p] - M/2) + n

def n_pq(p, q):
    p-=1
    return q - np.floor(chi[p] - M/2) 

def L_np(n, p, x):
    tmp = 1
    for q in range(1, M+1):
        if n != q:
            tmp *= (x - o_np(q, p))/(o_np(n, p) - o_np(q, p))
    return tmp

def G(xp, yp):
    return 1j*hankel1(0, k*np.linalg.norm(xp - yp))/4

def make_b(x, dx):
    N = len(x)
    h = 2*np.pi/N
    B = np.zeros((N, N), dtype=np.complex128)
    
    # construct B matrix
    for i in range(N):
        for j in range(N):
            ds = np.linalg.norm(dx[j])*h
            if np.abs(ll(i, j, N)) >=a:
                B[i, j] = ds*G(x[i], x[j])
    return B

def make_c(x, dx):
    N = len(x)
    h = 2*np.pi/N
    C = np.zeros((N, N), dtype=np.complex128)
    
    #Lmat = [[L_np(n_pq(p, d), p, chi[p-1]) for p in range(1, 2*m+1)] for d in np.arange(-N//2, N//2 + 1)]
    Omat = [[o_np(n, p) for n in range(1, M+1)] for p in range(1, 2*m+1)]
    
    for i in range(N):
        for p in range(1, 2*m+1):
            Op = Omat[p-1] 
            #L   = lagrangeInterp(ind, xx(xidx)); 
            for dist in Op:
                idx = i + dist
                if idx >= N:
                    idx = N - idx
                idx = int(idx)
                ds = np.linalg.norm(dx[idx])*h
                C[i, idx] += w[p-1]*G(0, chi[p-1]*h)*L_np(n_pq(p, dist), p, chi[p-1])*ds
    
    return C
    
def coeff_matrix(x, dx):
    B = make_b(x, dx)
    C = make_c(x, dx)
    return B + C

lmat = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        lmat[i, j] = ll(i, j, 7)
print(lmat)

#%%
#boundary curves
def circ(theta):
    return np.array([np.cos(theta), np.sin(theta)])
    # r = 9/20 - (1/9)*np.cos(5*theta)
    # return r*np.array([np.cos(theta), np.sin(theta)])
    
def dcirc(theta):
    return np.array([-np.sin(theta), np.cos(theta)])
    # r = 9/20 - (1/9)*np.cos(5*theta)
    # dr = (5/9)*np.sin(5*theta)
    # fdg = dr*np.array([np.cos(theta), np.sin(theta)])
    # gdf = r*np.array([-np.sin(theta), np.cos(theta)])
    # return fdg + gdf

def run1(N):
    Np = N
    angs = np.linspace(0, 2*np.pi, Np, endpoint=False)
    
    #discretize boundary
    points = [circ(t) for t in angs]
    dpoints = [dcirc(t) for t in angs]
    
    #coefficient matrix
    B = make_b(points, dpoints)
    C = make_c(points, dpoints)
    
    A = B + C
    
    #rhs data
    def source(x):
        return G(x, np.array([0, 0.5])) + 2*G(x, np.array([0, -0.5]))
            
    rhs = [source(circ(t)) for t in angs]
    
    #solve the system
    sol = np.linalg.solve(A, rhs)
    
    #solve the exterior helmholtz problem by integration
    measureangs = np.linspace(0, 2*np.pi, num=7, endpoint=False)
    measurepts = 3*np.array([[np.cos(a), np.sin(a)] for a in measureangs])
    
    def usc(xp):
        yfn = [G(xp, points[i])*sol[i]*np.linalg.norm(dpoints[i]) for i in range(Np)]
        h = angs[1]
        tmp = 0j
        for i in range(Np):
            tmp += h*yfn[i]
        return tmp
        
    alpertsol = np.array([usc(xp) for xp in measurepts])
    actual = np.array([source(xp) for xp in measurepts])
    
    err = np.abs(actual - alpertsol)
    print(N)
    print(actual)
    print(alpertsol)
    print(err)
    print(np.max(err)/np.max(np.abs(actual)))
    print()
    return err

run1(2**5)

#sweep Np
pows = np.arange(5,11)
errv = []
for N in 2**pows:
    err = run1(N)
    errv += [err]

#%%
plt.loglog(2*np.pi/2**pows, errv, '.-')
#plt.loglog(2*np.pi/2**pows, np.abs(np.log2((2*np.pi/2**pows)))*(2*np.pi/2**pows)**6, 'k--')
plt.show()