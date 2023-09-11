import numpy as np
from scipy.special import hankel1
from scipy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt

#time
N = 100
x = np.linspace(-2, 2, num=N+1)
y = x
dx = x[1]-x[0]
T = 4
M = 2**10

t = np.linspace(0, T, endpoint=False, num=M)
dt = T/M

print(dt/dx < 0.5)
print(dx, dt)

u_sc = np.zeros((len(x), len(y), len(t)))

X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

a = dt/dx
b = (a-1)/(a+1)

def srcfn(t):
    return np.exp(-(t-1)**2/0.01)

force = np.zeros(u_sc.shape)
for i in range(len(t)):
    force[N//2, N//2, i]=srcfn(t[i])

#center of point source
r0 = (0, 0)
#find the index

for i in range(2, len(t)):
    
    #update the rest of the scattered field
    u_sc[1:-1, 1:-1, i] = 2*u_sc[1:-1, 1:-1, i-1]  - u_sc[1:-1, 1:-1, i-2] + a**2*(u_sc[:-2, 1:-1, i-1] + u_sc[1:-1, :-2, i-1] +
        u_sc[2:, 1:-1, i-1] + u_sc[1:-1, 2:, i-1] - 4*u_sc[1:-1, 1:-1, i-1]) + dt**2*force[1:-1, 1:-1, i-1]
    
    #print(max(np.abs(u_sc[0, 1:, i])))
    #apply absorbing boundary conditions
    #left boundary
    u_sc[0, 1:-1, i] = b*(u_sc[1, 1:-1, i] - u_sc[0, 1:-1, i-1]) + u_sc[1, 1:-1, i-1]

    #right 
    u_sc[-1, 1:-1, i] = b*(u_sc[-2, 1:-1, i] - u_sc[-1, 1:-1, i-1]) + u_sc[-2, 1:-1, i-1]

    #bottom
    u_sc[1:-1, 0, i] = b*(u_sc[1:-1, 1, i] - u_sc[1:-1, 0, i-1]) + u_sc[1:-1, 1, i-1]

    #top
    u_sc[1:-1, -1, i] = b*(u_sc[1:-1, -2, i] - u_sc[1:-1, -1, i-1]) + u_sc[1:-1, -2, i-1]

#%%
def ptsrc(x, y, r0, t, g):
    G = ifft(g(t))
    dt = t[1] - t[0]
    f = fftfreq(len(t), d=dt)
    r = np.sqrt((x - r0[0])**2 + (y-r0[1])**2)
    u = 1j*np.zeros((*x.shape, len(t)))
    for i in range(1,len(f)//2):
        u[:, :, i] = 1j*G[i]*hankel1(0, 2*np.pi*f[i]*r)/4
        u[:, :, len(t) - i] = u[:, :, i].conj()
    u[:, :, 0] = G[0]*np.log(r)/(2*np.pi)
    return fft(u)

u = ptsrc(X, Y, [0,0], t, srcfn)

#%%
tidx = 384
fig, ax = plt.subplots(1, 3)

im = ax[0].matshow(np.abs(u_sc[:,:,tidx]), 
                 extent=(min(x), max(x), min(x), max(x)))
plt.colorbar(im)
im = ax[1].matshow(np.abs(u[:,:,tidx]), 
                 extent=(min(x), max(x), min(x), max(x)))
plt.colorbar(im)
im = ax[2].matshow(np.log10(np.abs(u_sc[:,:,tidx] - u[:,:,tidx])), 
                 extent=(min(x), max(x), min(x), max(x)))
plt.colorbar(im)
#im = plt.matshow(u_sc[:,:,300], extent=(min(x), max(x), min(x), max(x)))
#im = plt.plot(t, u[0,0, :])

plt.xlabel('x')
plt.ylabel('y')
plt.show()
