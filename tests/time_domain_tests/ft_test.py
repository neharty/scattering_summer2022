import numpy as np
from scipy.fft import fft, fftn, fftfreq, 
import matplotlib.pyplot as plt

from Ncyl.tds import cyl, scattering, time_domain_scattering
from utils import example_setups

#sigma=0.5
#gaussian_deriv = lambda x, y, t : np.sqrt(2/sigma)*(x+t-10)*np.exp(0.5-((x+t-10)**2)/sigma)

cyls = [cyl('d', [0, 0], 1)]
times = np.linspace(0, 30, num=100)
x = np.linspace(-5, 5, num = 300)
y = np.copy(x)
X, Y, T = np.meshgrid(x, y, times)

angs = np.linspace(0, 2*np.pi, num=100)
circ = np.array([[np.cos(ang), np.sin(ang)] for ang in angs])

for t in range(len(times)):
    plt.clf()
    plt.imshow(U[:, :, t], origin='upper', extent=(x.min(), x.max(), y.min(), y.max()))
    plt.plot(circ[:, 0], circ[:, 1], 'r')
    plt.title('t = ' + str(times[t]))
    plt.tight_layout()
    plt.colorbar()
    plt.clim(m, M)
    plt.savefig('deriv_of_gaussian/'+str(t)+'.png')

