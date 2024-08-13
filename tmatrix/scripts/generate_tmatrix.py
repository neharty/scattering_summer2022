# try to solve using alpert quadrature
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from integrators.alpert import alpert2, alpert6, alpert10, alpert16
from scatterers import scat_select
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
    impedance: object = None
    
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
            
        self.scgeom = scat_select(self.geometry, self.bc, [0, 0], self.radius, self.rotation, impedance=self.impedance)
        self.use_weighted = self.scgeom.weighted_density
                   
        if self.scgeom.open_arc and self.bc == 'n':
                self.t = np.linspace(0, 2*np.pi, 2*self.Npoints, False) #points are j*pi/Npoints
                self.h = np.pi/self.Npoints     
        else:
            self.h = 2*np.pi/self.Npoints  
            if self.scgeom.offset_t:
                self.t = (np.arange(1, self.Npoints+1) - 0.5)*self.h
            else:
                self.t = np.linspace(0, 2*np.pi, self.Npoints, False)

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
    
    def dphihatstardn(self, n, x, dx):
        r, theta = self.cart_to_polar(x)
        nv = np.array([dx[1], -dx[0]])
        nv /= np.linalg.norm(nv)
        xhat = x/np.linalg.norm(x)
        drdn = nv.dot(xhat)
        dmodedn = -nv.dot(np.array([-x[1], x[0]])*1j*n/(x[0]**2 + x[1]**2))
        return self.k*jvp(n, self.k*np.linalg.norm(x))*drdn*np.exp(-1j*n*theta) + self.phihatstar(n, x)*dmodedn
        
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
    
    def G0(self, xp, yp):
        return -np.log(np.linalg.norm(xp - yp))/(2*np.pi)
    
    def G(self, xp, yp):
        return 1j*hankel1(0, self.k*np.linalg.norm(xp - yp))/4
    
    def dGdn(self, xp, yp, nvect):
        v = xp-yp
        nrm = np.linalg.norm(v)
        dg = v.dot(nvect)/nrm
        tmp = -self.k*1j*hankel1(1, self.k*nrm)/4
        return tmp*dg
    
    def K(self, s1, s2):
        g1 = self.scgeom.bdy(s1)
        g2 = self.scgeom.bdy(s2)
        dg1 = self.scgeom.bdy(s1)
        dg2 = self.scgeom.bdy(s2)
        n1 = np.array([dg1[1], -dg1[0]])
        n1 = n1/np.linalg.norm(n1)
        n2 = np.array([dg2[1], -dg2[0]])
        n2 = n2/np.linalg.norm(n1)
        return -1j*self.k**2*hankel1(0, self.k*np.linalg.norm(g1-g2))*((g1-g2).dot(n1)*(g1-g2).dot(n2))/np.linalg.norm(g1-g2)**2/4 \
            +(1j*self.k*hankel1(1, self.k*np.linalg.norm(g1-g2))/4 - 1/(2*np.pi))\
                *(((2*(g1-g2).dot(n1)*(g1-g2).dot(n2))/np.linalg.norm(g1-g2)**4) - n1.dot(n2)/np.linalg.norm(g1-g2)**2)
    
    def K0(self, s1, s2):
        if np.cos(s1) == np.cos(s2):
            g = self.scgeom.bdy(s1)
            dg = self.scgeom.dbdy(s1)
            ddg = self.scgeom.ddbdy(s1)
            dddg = self.scgeom.dddbdy(s1)
            return -((dg.dot(ddg)**2)/(dg.dot(dg))**2 + ddg.dot(ddg)/(2*dg.dot(dg)) + dg.dot(dddg)/(3*(dg.dot(dg)))) - 1/6
        else:
            g1 = self.scgeom.bdy(s1)
            g2 = self.scgeom.bdy(s2)
            dg1 = self.scgeom.dbdy(s1)
            dg2 = self.scgeom.dbdy(s2)
            Ntilde = ((g1 - g2).dot(dg1)*(g1-g2).dot(dg2))/(np.linalg.norm(g1 - g2)**2)
            return -2*(dg1.dot(dg2) - 2*Ntilde)/(np.linalg.norm(g1 - g2)**2) - 1/(2*(np.sin((s1 - s2)/2))**2)
    
    def H(self, s1, s2):
        return np.sin(s1)*np.sin(s2)*self.K(s1, s2)
    
    def H0(self, s1, s2):
        return np.sin(s1)*np.sin(s2)*self.K0(s1, s2)
    
    def make_b_dirichlet(self):
        x = np.array([self.scgeom.bdy(ti) for ti in self.t])
        dx = np.array([self.scgeom.dbdy(ti) for ti  in self.t], dtype=np.complex128)
        B = np.array([[self.G(xp, yp) for yp in x] for xp in x])
    
        # construct B matrix
        for i in range(self.Npoints):
            for j in range(self.Npoints):
                if np.abs(self.ll(i, j)) < self.bie.dist:
                    B[i, j] = 0
        
        if not self.use_weighted:
            B = B @ np.diag(np.linalg.norm(dx, axis=1))
        
        B *= self.h
        return B
    
    def make_c_dirichlet(self):
        x = np.array([self.scgeom.bdy(ti) for ti in self.t])
        dx = np.array([self.scgeom.dbdy(ti) for ti in self.t])
        
        C = np.zeros((self.Npoints, self.Npoints), dtype=np.complex128)
        
        m = len(self.bie.nodemat)
        xalp = np.array([[self.scgeom.bdy(ti + self.bie.nodemat[p]*self.h) for ti in self.t] for p in range(m)])
        dxalp = np.array([[self.scgeom.dbdy(ti + self.bie.nodemat[p]*self.h) for ti in self.t] for p in range(m)])
        
        for p in range(m):
            Op = np.floor(self.bie.nodemat[p] - self.bie.M/2 + 1) + np.arange(0, self.bie.M)
            Lmat = self.L_np(Op, self.bie.nodemat[p])
            for i in range(self.Npoints):
                for dist in Op:
                    idx = i + dist
                    if idx >= self.Npoints:
                        idx = idx - self.Npoints
                    idx = int(idx)
                    kervaltmp = self.G(x[i], xalp[p][i])
                    if self.use_weighted:
                        kerval = kervaltmp*self.h
                    else:
                        ds = np.linalg.norm(dxalp[p][i])
                        kerval = kervaltmp*ds*self.h
                    lidx = np.where(Op == dist)[0][0]
                    C[i, idx] += self.bie.weightmat[p]*Lmat[lidx]*kerval
        return C
    
    def make_b_neumann(self):
        x = np.array([self.scgeom.bdy(ti) for ti in self.t])
        dx = np.array([self.scgeom.dbdy(ti) for ti  in self.t], dtype=np.complex128)
        nv = np.array([[xp[1], -xp[0]] for xp in dx])
        if not self.use_weighted:
            nv = nv.T/np.array([np.linalg.norm(n) for n in nv]) # normalize vectors
            nv = nv.T
        B = np.array([[self.dGdn(xp, yp, n) for yp in x] for (xp, n) in zip(x, nv)])
    
        # construct B matrix
        for i in range(self.Npoints):
            for j in range(self.Npoints):
                if np.abs(self.ll(i, j)) < self.bie.dist:
                    B[i, j] = 0
                    
        if not self.use_weighted:
            B = B @ np.diag(np.linalg.norm(dx, axis=1))
            
        B *= self.h
        return B
    
    def make_c_neumann(self):
        x = np.array([self.scgeom.bdy(ti) for ti in self.t])
        dx = np.array([self.scgeom.dbdy(ti) for ti in self.t])
        nv = np.array([[xp[1], -xp[0]] for xp in dx])
        if not self.use_weighted:
            nv = nv.T/np.array([np.linalg.norm(n) for n in nv]) # normalize vectors
            nv = nv.T
        C = np.zeros((self.Npoints, self.Npoints), dtype=np.complex128)
        
        m = len(self.bie.nodemat)
        xalp = np.array([[self.scgeom.bdy(ti + self.bie.nodemat[p]*self.h) for ti in self.t] for p in range(m)])
        dxalp = np.array([[self.scgeom.dbdy(ti + self.bie.nodemat[p]*self.h) for ti in self.t] for p in range(m)])
        
        for p in range(m):
            Op = np.floor(self.bie.nodemat[p] - self.bie.M/2 + 1) + np.arange(0, self.bie.M)
            Lmat = self.L_np(Op, self.bie.nodemat[p])
            for i in range(self.Npoints):
                for dist in Op:
                    idx = i + dist
                    if idx >= self.Npoints:
                        idx = idx - self.Npoints
                    idx = int(idx)
                    kervaltmp = self.dGdn(x[i], xalp[p][i], nv[i])
                    if self.use_weighted:
                        kerval = kervaltmp*self.h
                    else:
                        ds = np.linalg.norm(dxalp[p][i])
                        kerval = kervaltmp*ds*self.h
                    lidx = np.where(Op == dist)[0][0]
                    C[i, idx] += self.bie.weightmat[p]*Lmat[lidx]*kerval
        return C
    
    def make_b_neumann_open_arc(self):
        x = np.array([self.scgeom.bdy(ti) for ti in self.t])
        dx = np.array([self.scgeom.dbdy(ti) for ti  in self.t], dtype=np.complex128)
        #B = np.array([[self.H(ti, tj) + self.H0(ti, tj) for ti in self.t] for tj in self.t])
        B = np.zeros((self.Npoints, self.Npoints), dtype=np.complex128)
        # construct B matrix
        for i in range(self.Npoints):
            for j in range(self.Npoints):
                if np.abs(self.ll(i, j)) < self.bie.dist:
                    B[i, j] = 0
                else:
                    B[i, j] = self.H(self.t[i], self.t[j]) + self.H0(self.t[i], self.t[j])
        B *= self.h
        return B/(2*np.pi)
    
    def make_c_neumann_open_arc(self):
        C = np.zeros((self.Npoints, self.Npoints), dtype=np.complex128)
        
        m = len(self.bie.nodemat)
        talp = np.array([[ti + self.bie.nodemat[p]*self.h for ti in self.t] for p in range(m)])
        kern = lambda s1, s2: self.H(s1, s2) + self.H0(s1, s2)
        
        for p in range(m):
            Op = np.floor(self.bie.nodemat[p] - self.bie.M/2 + 1) + np.arange(0, self.bie.M)
            Lmat = self.L_np(Op, self.bie.nodemat[p])
            for i in range(self.Npoints):
                for dist in Op:
                    idx = i + dist
                    if idx >= self.Npoints:
                        idx = idx - self.Npoints
                    idx = int(idx)
                    kervaltmp = kern(self.t[i], talp[p][i])
                    kerval = kervaltmp*self.h
                    lidx = np.where(Op == dist)[0][0]
                    C[i, idx] += self.bie.weightmat[p]*Lmat[lidx]*kerval
        return C/(2*np.pi)
    
    def make_d_neumann_open_arc(self):
        tmp = np.zeros((self.Npoints, self.Npoints), dtype=np.complex128)
        for i in range(self.Npoints):
            if i==0:
                tmp[:, i] = -self.Npoints/2
            elif i%2==1:
                tmp[:, i] = 1/(2*self.Npoints*(np.sin(i*np.pi/(2*self.Npoints)))**2)
            elif i%2==0:
                tmp[:, i] = 0
        return tmp
        
    def coeff_matrix(self):
        if self.bc == 'd':
            B = self.make_b_dirichlet()
            C = self.make_c_dirichlet()
            return B + C
        if self.bc == 'n':
            if self.scgeom.open_arc:
                tmp = self.make_b_neumann_open_arc() + self.make_c_neumann_open_arc() + self.make_d_neumann_open_arc()
                return tmp
            else:
                B = self.make_b_neumann()
                C = self.make_c_neumann()
                return -np.eye(*B.shape)/2 + B + C
        if self.bc == 'i':
            # open arc can't be implemented
            Bd = self.make_b_dirichlet()
            Cd = self.make_c_dirichlet()
            Bn = self.make_b_neumann()
            Cn = self.make_c_neumann()
            return -np.eye(*Bn.shape)/2 + Bn + Cn - 1j*self.k*self.impedance(self.k)*(Bd + Cd)
  
    def Tnm(self, n, mu):
        #trapezoidal rule
        if self.scgeom.open_arc and self.bc =='n':
            self.t = np.linspace(0, 2*np.pi, 2*self.Npoints, False)
            yfn = np.array([self.phihatstar(n, self.scgeom.bdy(self.t[i]))*mu[i]*np.linalg.norm(self.scgeom.dbdy(self.t[i])) for i in range(len(mu))])
        else:
            if self.use_weighted:
                yfn = np.array([self.phihatstar(n, self.scgeom.bdy(self.t[i]))*mu[i] for i in range(len(mu))])
            else:
                yfn = np.array([self.phihatstar(n, self.scgeom.bdy(self.t[i]))*mu[i]*np.linalg.norm(self.scgeom.dbdy(self.t[i])) for i in range(len(mu))])
        tmp = 0j
        for i in range(self.Npoints):
            tmp += yfn[i]
        tmp *= self.h
        return tmp
    
    def neumann_rhs_open_arc(self, n, s):
        dpsidn = self.neumann_rhs(n, self.scgeom.bdy(s), self.scgeom.dbdy(s))
        if s == 0 or s == np.pi:
            return 0j
        if s > np.pi:
            return -np.linalg.norm(self.scgeom.dbdy(s))*np.sin(s)*dpsidn
        else:
            return np.linalg.norm(self.scgeom.dbdy(s))*np.sin(s)*dpsidn
        
    def neumann_rhs(self, n, x, dx):
        r, theta = self.cart_to_polar(x)
        nv = np.array([dx[1], -dx[0]])
        nv /= np.linalg.norm(nv)
        xhat = x/np.linalg.norm(x)
        drdn = nv.dot(xhat)
        dmodedn = nv.dot(np.array([-x[1], x[0]])*1j*n/(x[0]**2 + x[1]**2))
        return self.k*jvp(n, self.k*np.linalg.norm(x))*drdn*np.exp(1j*n*theta) + self.phihat(n, x)*dmodedn
    
    def make_rhs(self, n):
        if self.scgeom.open_arc and self.bc == 'n':
            return -np.array([self.neumann_rhs_open_arc(n, t) for t in self.t[:self.Npoints]])
        else:
            if self.bc == 'd':
                 return -np.array([self.phihat(n, self.scgeom.bdy(self.t[i])) for i in range(self.Npoints)])
            elif self.bc == 'n':
                if self.use_weighted:
                    return -np.array([self.neumann_rhs(n, self.scgeom.bdy(self.t[i]), self.scgeom.dbdy(self.t[i]))*np.linalg.norm(self.scgeom.dbdy(self.t[i])) for i in range(self.Npoints)])
                else:
                    return -np.array([self.neumann_rhs(n, self.scgeom.bdy(self.t[i]), self.scgeom.dbdy(self.t[i])) for i in range(self.Npoints)])
            elif self.bc == 'i':
                dirpart = np.array([self.phihat(n, self.scgeom.bdy(self.t[i])) for i in range(self.Npoints)])
                if self.use_weighted:
                    neumpart = -np.array([self.neumann_rhs(n, self.scgeom.bdy(self.t[i]), self.scgeom.dbdy(self.t[i]))*np.linalg.norm(self.scgeom.dbdy(self.t[i])) for i in range(self.Npoints)])
                else:
                    neumpart = -np.array([self.neumann_rhs(n, self.scgeom.bdy(self.t[i]), self.scgeom.dbdy(self.t[i])) for i in range(self.Npoints)])
                return -(neumpart - 1j*self.k*self.scgeom.impedance(self.k)*dirpart)
    
    def make_density(self):
        idxarr = np.arange(-self.Nmodes, self.Nmodes+1)
        mumat = []
        
        A = self.coeff_matrix()
        
        for n in idxarr:
            tmp = np.linalg.solve(A, self.make_rhs(n))
            if self.scgeom.open_arc and self.bc =='n':
                tmp = np.concatenate((tmp, -np.flip(tmp)))
            mumat += [tmp]
        return mumat
    
    def make_tmatrix(self):
        M = 2*self.Nmodes+1
        idxarr = np.arange(-self.Nmodes, self.Nmodes+1)
        T = np.zeros((M, M), dtype=np.complex128)
        
        mumat = self.make_density()
        
        for n in range(M):
            for m in range(M):
                # compute the T matrix entries using trapezoidal rule
                T[n, m] = 1j*self.Tnm(idxarr[n], mumat[m])/4
        
        return T

