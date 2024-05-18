# try to solve using alpert quadrature
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from integrators.alpert import alpert2, alpert6, alpert10, alpert16
#from Ncyl.cqm import scattering

bie = alpert6()

# wavenumber
k = 2 + 2j

def ll(i, j, N):
    # from Erli's code
    a = np.mod(i-j, N)
    b = np.mod(j-i, N)
    if a<b:
        return -a
    else:
        return b

def L_np(Op, x):
    tmp = np.ones(len(Op))
    for i in range(len(Op)):
        for j in range(len(Op)):
            if i != j:
                tmp[i] *= (x - Op[j])/(Op[i] - Op[j])
    return tmp

def G(xp, yp):
    return 1j*hankel1(0, k*np.linalg.norm(xp - yp))/4

def make_b(xfn, dxfn, N):
    t = np.linspace(0, 2*np.pi, N, False)
    x = np.array([xfn(ti) for ti in t])
    dx = np.array([dxfn(ti) for ti  in t], dtype=np.complex128)
    
    h = 2*np.pi/N
    B = np.array([[G(xp, yp) for yp in x] for xp in x])

    # construct B matrix
    for i in range(N):
        for j in range(N):
            if np.abs(ll(i, j, N)) < bie.dist:
                B[i, j] = 0
    
    B = B @ np.diag(np.linalg.norm(dx, axis=1))
    B *=h
    return B

def make_c(xfn, dxfn, N):
    t = np.linspace(0, 2*np.pi, N, False)
    x = np.array([xfn(ti) for ti in t])
    dx = np.array([dxfn(ti) for ti in t])
    
    h = 2*np.pi/N
    C = np.zeros((N, N), dtype=np.complex128)
    
    m = len(bie.nodemat)
    xalp = np.array([[xfn(ti + bie.nodemat[p]*h) for ti in t] for p in range(m)])
    dxalp = np.array([[dxfn(ti + bie.nodemat[p]*h) for ti in t] for p in range(m)])
    
    for p in range(m):
        Op = np.floor(bie.nodemat[p] - bie.M/2 + 1) + np.arange(0, bie.M)
        Lmat = L_np(Op, bie.nodemat[p])
        for i in range(N):
            for dist in Op:
                idx = i + dist
                if idx >= N:
                    # print(idx)
                    idx = idx - N
                    # print(idx)
                idx = int(idx)
                ds = np.linalg.norm(dxalp[p][i])
                kervaltmp = G(x[i], xalp[p][i])
                kerval = kervaltmp*ds*h
                C[i, idx] += bie.weightmat[p]*Lmat[np.where(Op == dist)]*kerval
    return C
    
def coeff_matrix(xfn, dxfn, N):
    B = make_b(xfn, dxfn, N)
    C = make_c(xfn, dxfn, N)
    return B + C

lmat = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        lmat[i, j] = ll(i, j, 7)
print(lmat)

#%%
#boundary curves
def circ(theta):
    # circle
    return np.array([np.cos(theta), np.sin(theta)])
    
    # starfish
    #r = 9/20 - (1/9)*np.cos(5*theta)
    #return r*np.array([np.cos(theta), np.sin(theta)])
    
    #star
    #r = 0.3
    #k = 5
    #return np.array([1.5*np.cos(theta) + (r/2)*np.cos((k+1)*theta) + (r/2)*np.cos((k-1)*theta),
    #        np.sin(theta) + (r/2)*np.sin((k+1)*theta) - (r/2)*np.sin((k-1)*theta)])
    
def dcirc(theta):
    # circle
    return np.array([-np.sin(theta), np.cos(theta)])

    # starfish
    #r = 9/20 - (1/9)*np.cos(5*theta)
    #dr = (5/9)*np.sin(5*theta)
    #fdg = dr*np.array([np.cos(theta), np.sin(theta)])
    #gdf = r*np.array([-np.sin(theta), np.cos(theta)])
    #return fdg + gdf
    
    #star
    #r = 0.3
    #k = 5
    #return np.array([-1.5*np.sin(theta) - (r/2)*(k+1)*np.sin((k+1)*theta) - (r/2)*(k-1)*np.sin((k-1)*theta),
    #        np.cos(theta) + (r/2)*(k+1)*np.cos((k+1)*theta) - (r/2)*(k-1)*np.cos((k-1)*theta)])

#rhs data
def source(x):
    return G(x, np.array([0, 0.5])) + 2*G(x, np.array([0, -0.5])) - 0.7*G(x, np.array([0.3, 0]))

#solve the exterior helmholtz problem by integration
measureangs = np.linspace(0, 2*np.pi, num=7, endpoint=False)
measurepts = 3*np.array([[np.cos(a), np.sin(a)] for a in measureangs])

def run1(N):
    Np = N
    angs = np.linspace(0, 2*np.pi, Np, endpoint=False)
    
    #coefficient matrix
    B = make_b(circ, dcirc, Np)
    C = make_c(circ, dcirc, Np)
    A = B + C
    
    rhs = [source(circ(t)) for t in angs]
    
    #solve the system
    sol = np.linalg.solve(A, rhs)
    
    def usc(xp):
        #trapezoidal rule
        yfn = [G(xp, circ(angs[i]))*sol[i]*np.linalg.norm(dcirc(angs[i])) for i in range(Np)]
        h = angs[1]
        tmp = 0j
        for i in range(Np):
            tmp += h*yfn[i]
        return tmp
        
    alpertsol = np.array([usc(xp) for xp in measurepts])
    actual = np.array([source(xp) for xp in measurepts])
    
    err = np.abs(actual - alpertsol)
    err = np.max(err)/np.max(np.abs(actual))
    print(N)
    print(actual)
    print(alpertsol)
    print(err)
    #print(np.max(err)/np.max(np.abs(actual)))
    print()
    return err

run1(2**5)

#sweep Np
pows = np.arange(5,11)
errvc = []
for N in 2**pows:
    err = run1(N)
    errvc += [err]

#%%
t = np.linspace(0, 2*np.pi, 1001)
pts = circ(t)
plt.plot(pts[0, :], pts[1, :])
plt.plot(measurepts[:, 0], measurepts[:, 1], 'kx')
plt.plot([0,0,0.3], [0.5, -0.5, 0], 'k*')
#plt.savefig('/home/nick/scattering/tests/bie_tests/alpert/starfish.pdf')
plt.show()

plt.loglog(2*np.pi/2**pows, errvc, '.-', label='infty-error')
plt.loglog(2*np.pi/2**pows, 1e-2*np.abs(np.log2((2*np.pi/2**pows)))*(2*np.pi/2**pows)**6, 'k--', label='log|h|h^16')
plt.xlabel('h')
plt.legend()
plt.tight_layout()
#plt.savefig('/home/nick/scattering/tests/bie_tests/alpert/error.pdf')
#plt.show()