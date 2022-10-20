import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp

def cart_to_polar(v):
        return (np.sqrt((v[0]**2 + v[1]**2)), np.arctan2(v[1], v[0]))

def polar_to_cart(v):
    return (v[0]*np.cos(v[1]), v[0]*np.sin(v[1]))

def make_direct_scattering_block(N):
        return np.eye(2*N + 1)

def S(m, n, v, k):
        # v is in cartesian coords
        return self.phi(m-n, v, k)

def j_smallarg(m, x):
    m = int(m)

    if m ==0:
        return 1
    else:
        if m < 0:
            return (-1)**(-m)*j_smallarg(-m, x)
        else: 
            return ((x/2)**m)/(np.math.factorial(m))

def h_smallarg(m, x):
    m = int(m)

    if m ==0:
        return 1+1j*(2*np.ln(x/2)/np.pi + np.euler_gamma)
    else:
        if m < 0:
            return (-1)**(-m)*h_smallarg(-m, x)
        else:
            return -1j/np.pi*np.math.factorial(m-1)*(x/2)**(-m)
        

def s_largearg(m, n, v, k):
    b, theta = cart_to_polar(v)
    w = m-n
    return np.sqrt(2/(np.pi*k*b))*np.exp(1j*(b - w*np.pi/2 - np.pi/4 + w*theta))

def phi(n, v, k):
        # v is in cartesian coords
        r, theta = self.cart_to_polar(v)
        return hankel1(n, k*r)*np.exp(1j*n*theta)

def make_mul_scattering_block(N, bc, radius, k, b, lam=None):

        # makes 1 block,i
        # cyl_i = incident cylinder,
        # cyl_sc = emitting cylinder (scattering from this incident on cyl_i)
        
        idxa_i = np.arange(-N, N + 1)
        idxa_sc = np.arange(-N, N + 1)

        for i in range(len(idxa_i)):
            for j in range(len(idxa_sc)):
                m, n = idxa_i[i], idxa_sc[j]
                if bc == 'd':
                    if m == 0:
                        sc[i, j] = 1
                    else:
                        scm[i,j] = j_smallarg*self.S(n, m, b)/hankel1(m, k*radius)
                elif bc == 'n':
                    scm[i,j] = jvp(m, k*radius)*self.S(n, m, b)/h1vp(m, k*radius)
                elif bc == 'i':
                    scm[i,j] = (k*jvp(m, k*radius) + lam*jv(m, k*radius))*self.S(n, m, b)/(k*h1vp(m, k*radius) + lam*hankel1(m, k*radius))
                else:
                    print('invalid bc')
                    break
        return scm
