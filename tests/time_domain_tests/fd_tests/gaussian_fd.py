import numpy as np
import waveforms.waveforms as wvf
import os

#time

N = 600
x = np.linspace(-6, 6, num=N+1)
y = np.linspace(-6, 6, num=N+1)
dx = x[1] - x[0]
dt = dx/2
T = 20
M = int(T/dt) + 1
print(M)
t = np.linspace(0, T, endpoint=False, num=M)

print(dx, dt)

u_sc = np.zeros((len(x), len(y), len(t)))

X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

wave = wvf.gaussian(time_shift=6)

inmask = (R >= 1-dx)
bdymask = (R >= (1-dx)) & (R <= (1+dx))
print(sum(bdymask))

a = dt/dx
b = (a-1)/(a+1)

for i in range(2, len(t)):
    #update incident field
    u_i = wave.td_wave(X, Y, t[i-1])
    
    #boundary condition
    u_sc[bdymask, i-1] = -u_i[bdymask]
    
       #update scattered field
    u_sc[1:-1, 1:-1, i] = 2*u_sc[1:-1, 1:-1, i-1]  - u_sc[1:-1, 1:-1, i-2] + a**2*(u_sc[:-2, 1:-1, i-1] + u_sc[1:-1, :-2, i-1] +
        u_sc[2:, 1:-1, i-1] + u_sc[1:-1, 2:, i-1] - 4*u_sc[1:-1, 1:-1, i-1])
    
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


    #field is zero inside the obstacle
    #u_sc[:, :, i] = inmask*u_sc[:, :, i]

if not os.path.exists('data'):
    os.makedirs('data')

xpts = [-5, -4, -3, -2, 2, 3, 4, 5]
for xd in xpts:
    idx = np.argwhere(x == xd)[0, 0]
    print(xd, idx)
    u_out = u_sc[N//2, idx, :]
    print(X[N//2, idx], Y[N//2, idx])
    arr = [[t[i], u_out[i]] for i in range(len(t))]
    np.savetxt('data/'+str(xd) + '_' + '0_.dat', arr)
