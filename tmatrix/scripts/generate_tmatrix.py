# try to solve using alpert quadrature
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from integrators.alpert import alpert2, alpert6, alpert10, alpert16
from scatterers import cyl, starfish, bean, leaf, fat_starfish
import argparse
import os
from dataclasses import dataclass

@dataclass
class Tmatrix:
    k: complex
    alpertorder: int
    geometry: str
    bc: str
    radius: float
    Npoints: int
    Nmodes: int
    rotation: float = 0.0
    
    def __post_init__(self):    
        self.bie = None
        if self.alpertorder == 2:
            self.bie = alpert2()
        if self.alpertorder == 6:
            self.bie = alpert6()
        if self.alpertorder == 10:
            self.bie = alpert10()
        if self.alpertorder == 16:
            self.bie = alpert16()
    
        self.t = np.linspace(0, 2*np.pi, self.Npoints, False)
        self.h = 2*np.pi/self.Npoints
        
        if self.geometry == 'cyl':
            self.scgeom = cyl(self.bc, [0,0], self.radius)
        elif self.geometry == 'starfish':
            self.scgeom = starfish(self.bc, [0,0], self.radius, self.rotation)
        elif self.geometry == 'bean':
            self.scgeom = bean(self.bc, [0, 0], self.radius, self.rotation)
        elif self.geometry == 'leaf':
            self.scgeom = leaf(self.bc, [0, 0], self.radius, self.rotation)
        elif self.geometry == 'fat_starfish':
            self.scgeom = fat_starfish(self.bc, [0,0], self.radius, self.rotation)
    
    def cart_to_polar(self, v):
        return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))
    
    def phihat(self, n, v):
        # v s in cartesian coords
        r, theta = self.cart_to_polar(v)
        return jv(n, self.k*r)*np.exp(1j*n*theta)
    
    def phihatstar(self, n, v):
        # v s in cartesian coords
        r, theta = self.cart_to_polar(v)
        return jv(n, self.k*r)*np.exp(-1j*n*theta)
    
    def ll(self, i, j):
        # from Erli's code
        a = np.mod(i-j, self.Npoints)
        b = np.mod(j-i, self.Npoints)
        if a<b:
            return -a
        else:
            return b
    
    def L_np(self, Op, x):
        tmp = np.ones(len(Op))
        for i in range(len(Op)):
            for j in range(len(Op)):
                if i != j:
                    tmp[i] *= (x - Op[j])/(Op[i] - Op[j])
        return tmp
    
    def G(self, xp, yp):
        return 1j*hankel1(0, self.k*np.linalg.norm(xp - yp))/4
    
    def dGdn(self, xp, yp, nvect):
        v = xp-yp
        nrm = np.linalg.norm(v)
        dg = v.dot(nvect)/nrm
        tmp = -self.k*1j*hankel1(1, self.k*nrm)/4
        return tmp*dg
    
    def make_b_dirichlet(self, xfn, dxfn):
        x = np.array([xfn(ti) for ti in self.t])
        dx = np.array([dxfn(ti) for ti  in self.t], dtype=np.complex128)
        B = np.array([[self.G(xp, yp) for yp in x] for xp in x])
    
        # construct B matrix
        for i in range(self.Npoints):
            for j in range(self.Npoints):
                if np.abs(self.ll(i, j)) < self.bie.dist:
                    B[i, j] = 0
        
        B = B @ np.diag(np.linalg.norm(dx, axis=1))
        B *= self.h
        return B
    
    def make_c_dirichlet(self, xfn, dxfn):
        x = np.array([xfn(ti) for ti in self.t])
        dx = np.array([dxfn(ti) for ti in self.t])
        
        C = np.zeros((self.Npoints, self.Npoints), dtype=np.complex128)
        
        m = len(self.bie.nodemat)
        xalp = np.array([[xfn(ti + self.bie.nodemat[p]*self.h) for ti in self.t] for p in range(m)])
        dxalp = np.array([[dxfn(ti + self.bie.nodemat[p]*self.h) for ti in self.t] for p in range(m)])
        
        for p in range(m):
            Op = np.floor(self.bie.nodemat[p] - self.bie.M/2 + 1) + np.arange(0, self.bie.M)
            Lmat = self.L_np(Op, self.bie.nodemat[p])
            for i in range(self.Npoints):
                for dist in Op:
                    idx = i + dist
                    if idx >= self.Npoints:
                        idx = idx - self.Npoints
                    idx = int(idx)
                    ds = np.linalg.norm(dxalp[p][i])
                    kervaltmp = self.G(x[i], xalp[p][i])
                    kerval = kervaltmp*ds*self.h
                    lidx = np.where(Op == dist)[0][0]
                    C[i, idx] += self.bie.weightmat[p]*Lmat[lidx]*kerval
        return C
    
    def make_b_neumann(self, xfn, dxfn):
        x = np.array([xfn(ti) for ti in self.t])
        dx = np.array([dxfn(ti) for ti  in self.t], dtype=np.complex128)
        nv = np.array([[xp[1], -xp[0]] for xp in dx])
        nv = nv.T/np.array([np.linalg.norm(n) for n in nv])#normalize vectors
        nv = nv.T
        B = np.array([[self.dGdn(xp, yp, n) for yp in x] for (xp, n) in zip(x, nv)])
    
        # construct B matrix
        for i in range(self.Npoints):
            for j in range(self.Npoints):
                if np.abs(self.ll(i, j)) < self.bie.dist:
                    B[i, j] = 0
        
        B = B @ np.diag(np.linalg.norm(dx, axis=1))
        B *= self.h
        return B
    
    def make_c_neumann(self, xfn, dxfn):
        x = np.array([xfn(ti) for ti in self.t])
        dx = np.array([dxfn(ti) for ti in self.t])
        nv = np.array([[xp[1], -xp[0]] for xp in dx])
        nv = nv.T/np.array([np.linalg.norm(n) for n in nv])#normalize vectors
        nv = nv.T
        C = np.zeros((self.Npoints, self.Npoints), dtype=np.complex128)
        
        m = len(self.bie.nodemat)
        xalp = np.array([[xfn(ti + self.bie.nodemat[p]*self.h) for ti in self.t] for p in range(m)])
        dxalp = np.array([[dxfn(ti + self.bie.nodemat[p]*self.h) for ti in self.t] for p in range(m)])
        
        for p in range(m):
            Op = np.floor(self.bie.nodemat[p] - self.bie.M/2 + 1) + np.arange(0, self.bie.M)
            Lmat = self.L_np(Op, self.bie.nodemat[p])
            for i in range(self.Npoints):
                for dist in Op:
                    idx = i + dist
                    if idx >= self.Npoints:
                        idx = idx - self.Npoints
                    idx = int(idx)
                    ds = np.linalg.norm(dxalp[p][i])
                    kervaltmp = self.dGdn(x[i], xalp[p][i], nv[i])
                    kerval = kervaltmp*ds*self.h
                    lidx = np.where(Op == dist)[0][0]
                    C[i, idx] += self.bie.weightmat[p]*Lmat[lidx]*kerval
        return C
    
    def coeff_matrix(self, xfn, dxfn):
        if self.bc == 'd':
            B = self.make_b_dirichlet(xfn, dxfn)
            C = self.make_c_dirichlet(xfn, dxfn)
            return B + C
        if self.bc == 'n':
            B = self.make_b_neumann(xfn, dxfn)
            C = self.make_c_neumann(xfn, dxfn)
            return -np.eye(*B.shape)/2 + B + C
  
    def Tnm(self, n, mu):
        #trapezoidal rule
        yfn = np.array([self.phihatstar(n, self.scgeom.bdy(self.t[i]))*mu[i]*np.linalg.norm(self.scgeom.dbdy(self.t[i])) for i in range(self.Npoints)])
        tmp = 0j
        for i in range(self.Npoints):
            tmp += yfn[i]
        tmp *= self.h
        return tmp
    
    def neumann_rhs(self, n, x, dx):
        r, theta = self.cart_to_polar(x)
        nv = np.array([dx[1], -dx[0]])
        nv /= np.linalg.norm(nv)
        xhat = x/np.linalg.norm(x)
        drdn = nv.dot(xhat)
        dmodedn = nv.dot(np.array([-x[1], x[0]])*1j*n/(x[0]**2 + x[1]**2))
        return self.k*jvp(n, self.k*np.linalg.norm(x))*drdn*np.exp(1j*n*theta) + self.phihat(n, x)*dmodedn
    
    def make_tmatrix(self):
        M = 2*self.Nmodes+1
        idxarr = np.arange(-self.Nmodes, self.Nmodes+1)
        T = np.zeros((M, M), dtype=np.complex128)
        mumat = []
        
        A = self.coeff_matrix(self.scgeom.bdy, self.scgeom.dbdy)
        
        for n in idxarr:
            # find the density for 1 mode
            if self.bc == 'd':    
                rhs = -np.array([self.phihat(n, self.scgeom.bdy(self.t[i])) for i in range(self.Npoints)])
            elif self.bc == 'n':
                rhs = -np.array([self.neumann_rhs(n, self.scgeom.bdy(self.t[i]), self.scgeom.dbdy(self.t[i])) for i in range(self.Npoints)])
            
            if np.isnan(np.abs(A)).any():
                mumat += [np.zeros(self.Npoints, dtype=np.complex128)]
            else:
                mumat += [np.linalg.solve(A, rhs)]
            
        for n in range(M):
            for m in range(M):
                # compute the T matrix entries using trapezoidal rule
                T[n, m] = 1j*self.Tnm(idxarr[n], mumat[m])/4
        
        return T

