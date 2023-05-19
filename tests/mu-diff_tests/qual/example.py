import numpy as np
import matplotlib.pyplot as plt
import pickle

from Ncyl.tds import scattering

rootdir = '/home/nick/scattering/tests/mu-diff_tests/qual/'

cyls = pickle.load(open(rootdir+'cyls.p', 'rb'))

sc = scattering(cyls, freq=1)
pts = np.linspace(-20, 20, num=401)
X, Y = np.meshgrid(pts, pts)

u = np.abs(sc.make_u(X, Y))

im = plt.imshow(u, extent=[-20, 20, -20, 20], origin='lower')
plt.xlabel('x')
plt.ylabel('y')
plt.title('example of total field solution')
plt.colorbar(im)
plt.axis('scaled')
plt.savefig('/home/nick/School/NJIT/research/qual_talk/example_u.png')



