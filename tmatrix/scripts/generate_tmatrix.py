# try to solve using alpert quadrature
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from integrators.alpert import alpert2, alpert4, alpert3, alpert6, alpert10, alpert16
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
        if self.alpertorder == 3:
            self.bie = alpert3()
        if self.alpertorder == 4:
            self.bie = alpert4()
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
        dmodedn = nv.dot(np.array([-x[1], x[0]])*(-1j*n)/(x[0]**2 + x[1]**2))
        return self.k*jvp(n, self.k*np.linalg.norm(x))*drdn*np.exp(-1j*n*theta) + self.phihatstar(n, x)*dmodedn
        
    def ll(self, i, j, N=0):
        if N == 0:
            N = self.Npoints
        # from Erli's code
        a = np.mod(i-j, N)
        b = np.mod(j-i, N)
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
    
    def K1(self, s1, s2):
        g1 = self.scgeom.bdy(s1)
        g2 = self.scgeom.bdy(s2)
        dg1 = self.scgeom.dbdy_neum_rhs(s1)
        dg2 = self.scgeom.dbdy_neum_rhs(s2)
        n1 = np.array([dg1[1], -dg1[0]])
        n1 = n1/np.linalg.norm(n1)
        n2 = np.array([dg2[1], -dg2[0]])
        n2 = n2/np.linalg.norm(n1)
        return -2*((-1j*self.k**2/4)*hankel1(0, self.k*np.linalg.norm(g1-g2))*((g1-g2).dot(n1)*(g1-g2).dot(n2))/np.linalg.norm(g1-g2)**2 \
            +((1j*self.k/4)*np.linalg.norm(g1-g2)*hankel1(1, self.k*np.linalg.norm(g1-g2)) - 1/(2*np.pi))\
                *(((2*(g1-g2).dot(n1)*(g1-g2).dot(n2))/np.linalg.norm(g1-g2)**4) - n1.dot(n2)/np.linalg.norm(g1-g2)**2))
    
    def K2(self, s1, s2):
        if abs(np.cos(s1) - np.cos(s2)) < 1e-12:
            g = self.scgeom.bdy(s1)
            dg = self.scgeom.dbdy(s1)
            ddg = self.scgeom.ddbdy(s1)
            dddg = self.scgeom.dddbdy(s1)
            return -0.5*(dg.dot(ddg))**2/np.linalg.norm(dg)**4 + 0.25*ddg.dot(ddg)/np.linalg.norm(dg)**2 + (1/6)*dg.dot(dddg)/np.linalg.norm(dg)**2
        else:
            g1 = self.scgeom.bdy(s1)
            g2 = self.scgeom.bdy(s2)
            dg1 = self.scgeom.dbdy_neum_rhs(s1)
            dg2 = self.scgeom.dbdy_neum_rhs(s2)
            return -dg1.dot(dg2)/np.linalg.norm(g1-g2)**2 + 2*((g1-g2).dot(dg1)*(g1-g2).dot(dg2))/np.linalg.norm(g1-g2)**4 - 1/(np.cos(s1)-np.cos(s2))**2
    
    def H1(self, s1, s2):
        return np.sin(s1)*np.sin(s2)*self.K1(s1, s2)
    
    def H2(self, s1, s2):
        tmp = self.K2(s1, s2)
        tmp1 = np.sin(s1)
        tmp2 = np.sin(s2)
        return tmp*tmp1*tmp2
    
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
        
    def make_k1b_neumann_open_arc(self):
        # construct B matrix
        Bact = np.zeros((self.Npoints-1, self.Npoints-1), dtype=np.complex128)
        for i in range(self.Npoints-1):
            for j in range(self.Npoints-1):
                if np.abs(self.ll(i, j, self.Npoints-1)) >= self.bie.dist:
                    Bact[i, j] = self.H1(self.t[i+1], self.t[j+1])
        Bact *= self.h
        return Bact
    
    def make_k1c_neumann_open_arc(self):
        C = np.zeros((self.Npoints-1, self.Npoints-1), dtype=np.complex128)
        m = len(self.bie.nodemat)
        talp = np.array([[ti + self.bie.nodemat[p]*self.h for ti in self.t] for p in range(m)])
        kern = lambda s1, s2: self.H1(s1, s2)
        
        for p in range(m):
            Op = np.floor(self.bie.nodemat[p] - self.bie.M/2 + 1) + np.arange(0, self.bie.M-1)
            Lmat = self.L_np(Op, self.bie.nodemat[p])
            for i in range(self.Npoints-1):
                for dist in Op:
                    idx = i + dist
                    if idx >= self.Npoints-1:
                        idx = idx - (self.Npoints-1)
                    idx = int(idx)
                    kerval = kern(self.t[i+1], talp[p, i+1])
                    lidx = np.where(Op == dist)[0][0]
                    C[i, idx] += self.bie.weightmat[p]*Lmat[lidx]*kerval
        
        return C*self.h
    
    def make_k2_neumann_open_arc(self):
        k2mat = np.array([[self.H2(ti, tj) for tj in self.t] for ti in self.t])
        ttmp = k2mat
        
        ddmat = np.zeros((self.Npoints-1, self.Npoints-1), dtype=np.complex128)
        for i in range(1, self.Npoints):
            for j in range(1, self.Npoints):
                ddmat[i-1, j-1] = ttmp[i,j] - ttmp[i, -j]
        
        return -ddmat*self.h/(2*np.pi)
        
    def make_d_neumann_open_arc(self):
        def tent(i):
            if i==0:
                return -self.Npoints/2
            elif i%2==1:
                return 1/(2*self.Npoints*((np.sin(i*np.pi/(2*self.Npoints))**2)))
            elif i%2==0:
                return 0
        
        tmp = np.zeros((2*self.Npoints, 2*self.Npoints), dtype=np.complex128)
        for i in range(2*self.Npoints):
            for j in range(2*self.Npoints):
                tmp[i, j] = tent(abs(i-j))
            
        #use odd 2pi periodicity
        d = np.zeros((self.Npoints-1, self.Npoints-1), dtype = np.complex128)
        for i in range(1, self.Npoints):
            for j in range(1, self.Npoints):
                d[i-1, j-1] = tmp[i, j] - tmp[i, -j]
                
        return d
    
    #---- impedance for open arcs ----
    
    def make_A11(self):
        
    
    def coeff_matrix(self):
        if self.bc == 'd':
            B = self.make_b_dirichlet()
            C = self.make_c_dirichlet()
            return B + C
        if self.bc == 'n':
            if self.scgeom.open_arc:
                k1b = self.make_k1b_neumann_open_arc()
                k1c = self.make_k1c_neumann_open_arc()
                k2 = self.make_k2_neumann_open_arc()
                d = self.make_d_neumann_open_arc()
                tmp = (k1b + k1c + k2 + d)
                return tmp
            else:
                B = self.make_b_neumann()
                C = self.make_c_neumann()
                return -np.eye(*B.shape)/2 + B + C
        if self.bc == 'i':
            if self.scgeom.open_arc:
                pass
            else:
                Bd = self.make_b_dirichlet()
                Cd = self.make_c_dirichlet()
                Bn = self.make_b_neumann()
                Cn = self.make_c_neumann()
                return -np.eye(*Bn.shape)/2 + Bn + Cn - 1j*self.k*self.impedance(self.k)*(Bd + Cd)
  
    def Tnm(self, n, mu):
        #trapezoidal rule
        if self.scgeom.open_arc and self.bc =='n':
            yfn = np.array([self.dphihatstardn(n, self.scgeom.bdy(self.t[i]), self.scgeom.dbdy_neum_rhs(self.t[i]))*mu[i-1]*np.linalg.norm(self.scgeom.dbdy(self.t[i])) for i in range(1, self.Npoints)])
        else:
            if self.use_weighted:
                yfn = np.array([self.phihatstar(n, self.scgeom.bdy(self.t[i]))*mu[i] for i in range(len(mu))])
            else:
                yfn = np.array([self.phihatstar(n, self.scgeom.bdy(self.t[i]))*mu[i]*np.linalg.norm(self.scgeom.dbdy(self.t[i])) for i in range(len(mu))])
        tmp = np.sum(yfn)
        tmp *= self.h
        return tmp
    
    def neumann_rhs_open_arc(self, n, s):
        dpsidn = self.neumann_rhs(n, self.scgeom.bdy(s), self.scgeom.dbdy_neum_rhs(np.cos(s)))
        return np.linalg.norm(self.scgeom.dbdy_neum_rhs(np.cos(s)))*np.sin(s)*dpsidn
        
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
            return -2*np.array([self.neumann_rhs_open_arc(n, t) for t in self.t[1:self.Npoints]])
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

