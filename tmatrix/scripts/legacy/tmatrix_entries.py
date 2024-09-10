# try to solve using alpert quadrature
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from integrators.alpert import alpert2, alpert6, alpert10, alpert16
from scatterers import cyl
import argparse
import os
from ast import literal_eval


parser = argparse.ArgumentParser()

parser.add_argument('alpertorder', type=int)
parser.add_argument('geometry')
parser.add_argument('kr', type=float)
parser.add_argument('ki', type=float)
parser.add_argument('Npoints', type=int)
parser.add_argument('Nmodes', type = int)
parser.add_argument('dout')

args = parser.parse_args()

if args.alpertorder == 2:
    bie = alpert2()
if args.alpertorder == 6:
    bie = alpert6()

# wavenumber
k = args.kr + args.ki*1j

N = args.Npoints
t = np.linspace(0, 2*np.pi, N, False)
h = 2*np.pi/N


scgeom = cyl('d', [0,0], 1)

def cart_to_polar(v):
    return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

def phihat(n, v):
    # v s in cartesian coords
    r, theta = cart_to_polar(v)
    return jv(n, k*r)*np.exp(1j*n*theta)

def phihatstar(n, v):
    # v s in cartesian coords
    r, theta = cart_to_polar(v)
    return jv(n, k*r)*np.exp(-1j*n*theta)

def ll(i, j):
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

def make_b(xfn, dxfn):
    x = np.array([xfn(ti) for ti in t])
    dx = np.array([dxfn(ti) for ti  in t], dtype=np.complex128)
    B = np.array([[G(xp, yp) for yp in x] for xp in x])

    # construct B matrix
    for i in range(N):
        for j in range(N):
            if np.abs(ll(i, j)) < bie.dist:
                B[i, j] = 0
    
    B = B @ np.diag(np.linalg.norm(dx, axis=1))
    B *= h
    return B

def make_c(xfn, dxfn):
    x = np.array([xfn(ti) for ti in t])
    dx = np.array([dxfn(ti) for ti in t])
    
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
                    idx = idx - N
                idx = int(idx)
                ds = np.linalg.norm(dxalp[p][i])
                kervaltmp = G(x[i], xalp[p][i])
                kerval = kervaltmp*ds*h
                lidx = np.where(Op == dist)[0][0]
                C[i, idx] += bie.weightmat[p]*Lmat[lidx]*kerval
    return C
    
def coeff_matrix(xfn, dxfn):
    B = make_b(xfn, dxfn)
    C = make_c(xfn, dxfn)
    return B + C

def Tnm(n, mu):
    #trapezoidal rule
    yfn = np.array([phihatstar(n, scgeom.bdy(t[i]))*mu[i]*np.linalg.norm(scgeom.dbdy(t[i])) for i in range(N)])
    tmp = 0j
    for i in range(N):
        tmp += yfn[i]
    tmp *= h
    return tmp

def main():
    M = 2*args.Nmodes+1
    idxarr = np.arange(-args.Nmodes, args.Nmodes+1)
    T = np.zeros((M, M), dtype=np.complex128)
    mumat = []
    
    A = coeff_matrix(scgeom.bdy, scgeom.dbdy)
    
    for n in idxarr:
        # find the density for 1 mode
        rhs = -np.array([phihat(n, scgeom.bdy(t[i])) for i in range(N)])
        mumat += [np.linalg.solve(A, rhs)]
        #print(mumat)
        tmp = np.max(np.abs(mumat[-1] + 1/((1j*np.pi/2)*hankel1(n, k*1))*np.exp(1j*n*t)))
        print(tmp)
        
    for n in range(M):
        for m in range(M):
            # compute the T matrix entries using trapezoidal rule
            T[n, m] = 1j*Tnm(idxarr[n], mumat[m])/4
    
    outstr = f'tmatrix_{args.geometry}_alpert{args.alpertorder}_kr{args.kr:.3f}_ki{args.ki:.3f}_Npoints{args.Npoints}_Nmodes{args.Nmodes}.dat'
    np.savetxt(os.path.join(args.dout, outstr), T)
    
if __name__ == "__main__":
    main()