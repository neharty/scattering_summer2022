import numpy as np
import matplotlib.pyplot as plt

sigma = 4 

def v(s, ti, tii):
    return (1/sigma - 0.5)*((ti + tii - 2*s)/(tii - ti))**3 + (2*s - ti - tii)/(sigma*(tii - ti)) + 0.5

def w(s, ti, tii):
    return (tii*v(s, ti, tii)**sigma + ti*(1 - v(s, ti, tii))**sigma)/(v(s, ti, tii)**sigma + (1-v(s, ti, tii))**sigma) 

t = np.linspace(0, 2*np.pi, 50)

# teardrop
a = 0.5
b = np.tan(a*np.pi/2)/2
x = lambda t: [np.sin(t/2) , -b*np.sin(t)]
plt.plot(*x(w(t, 0, 2*np.pi)), '.-')
plt.axis('equal')
plt.show()

# boomerang
def boomx(t):
    return np.array([[-(2/3)*np.sin(3*ti/2), np.sin(ti)] for ti in t])
plt.plot(boomx(w(t, 0, 2*np.pi))[:, 0], boomx(w(t, 0, 2*np.pi))[:, 1], '.-')
plt.axis('equal')
plt.show()

# strip
def stripx(t):
    return np.array([[ti/np.pi - 1, 0] for ti in t])
plt.plot(stripx(w(t, 0, 2*np.pi))[:, 0], stripx(w(t, 0, 2*np.pi))[:, 1], '.-')
plt.axis('equal')
plt.show()

# v strip
def vstripx1(t):
    #split into 2 parts
    return np.array([[ti/(np.pi) - 1, 1-ti/(np.pi)] for ti in t])

def vstripx2(t):
    return np.array([[ti/np.pi - 1, ti/np.pi - 1] for ti in t])

def vstripx(t):
    mask1 = t <= np.pi
    mask2 = t >= np.pi
    tmp = np.vstack((vstripx1(w(t[mask1], 0, np.pi)), vstripx2(w(t[mask2], np.pi, 2*np.pi))))
    return tmp

plt.plot(vstripx(t)[:, 0], vstripx(t)[:, 1], '.-')
plt.axis('equal')
plt.show()
