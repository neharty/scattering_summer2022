import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from Ncyl.cqm import scattering

gamma_dict = {1 : 4.967362978287758,
            2 :-16.20501504859126,
            3 : 25.85153761832639,
            4 : -22.22599466791883,
            5 : 9.930104998037539,
            6 : -1.817995878141594}

k = 1

def circ(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def dcirc(theta):
    return np.array([-np.sin(theta), np.cos(theta)])

def l(i, j, N):
    tmp = np.abs(np.mod(i-j, N))
    if tmp > N//2:
        tmp = N-tmp
    return tmp

def coeff_matrix(x, dx):
    N = len(x)
    
    def G(xp, yp):
        return 1j*hankel1(0, k*np.linalg.norm(xp - yp))/4
    
    h = 2*np.pi/(N-1)
    A = np.zeros((N, N), dtype=np.complex128)
    
    for im in range(N):
        for jm in range(N):
            i = im + 1
            j = jm + 1
            if im == jm:
                A[im, jm] = 0
            else:
                ds = np.linalg.norm(dx[jm])*h
                if l(i, j, N) <= 6:
                    A[im, jm] = ds*(1 + gamma_dict[l(i, j, N)])*G(x[im], x[jm])
                else:
                    A[im, jm] = ds*G(x[im], x[jm])
    return A

nl = 8
lmat = np.zeros((nl, nl))
for i in range(nl):
    for j in range(nl):
        lmat[i, j] = l(i+1, j+1, nl)
print(lmat)

angs = np.linspace(0, 2*np.pi, 401)
points = [circ(t) for t in angs]
dpoints = [dcirc(t) for t in angs]
A = coeff_matrix(points, dpoints)
plt.matshow(np.abs(A))
plt.show()

n = 1
rhs = [-jv(n, k*np.linalg.norm(circ(t)))*np.exp(1j*n*t) for t in angs]
sol = np.linalg.solve(A, rhs)

#%%
plt.plot(angs, sol.real, label=r'real $\mu$ KR')
plt.plot(angs, sol.imag, label=r'imag $\mu$ KR')
mu = 4*np.exp(1j*n*angs)/(1j*(-1)**n*hankel1(n, k*1)*2*np.pi)
plt.plot(angs, mu.real, label=r'real $\mu$ exact')
plt.plot(angs, mu.imag, label=r'imag $\mu$ exact')
plt.legend()
plt.xlabel(r'$\theta$')