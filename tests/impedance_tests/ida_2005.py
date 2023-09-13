import numpy as np
import matplotlib.pyplot as plt
from Ncyl.cqm import cyl, time_domain_scattering
from waveforms.waveforms import gauss_mod_sine_pulse
from impedance import ida_ishihara_goto
from integrators.integrators import IRK3

t = np.linspace(1, 2, 200)
phi = np.arcsin(0.6/0.78)
wv = gauss_mod_sine_pulse(time_shift=1.5,
                          inc_angle=np.pi-phi)
#plt.plot(t, wv.profile(t).real)

cyls = [cyl('i', [0,0], 0.6, impedance=ida_ishihara_goto)]

r = 1.2*0.6
ang = 120*np.pi/180
x, y = (r*np.cos(ang), r*np.sin(ang))
print(x, y)

X, Y = np.meshgrid([x], [y])

tint = None
tds = time_domain_scattering(cyls, 
                             wv, 
                             7, 
                             sampnum = 2**10, 
                             modes=64,
                             scargs={'method':'explicit'}
                             )
u = tds.make_U_sc(X, Y)
plt.plot(tds.times, u[0,0,:])

