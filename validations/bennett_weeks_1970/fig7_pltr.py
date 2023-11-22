import numpy as np
import matplotlib.pyplot as plt

dpath = '/home/nick/scattering/validations/bennett_weeks_1970/'
oms = np.loadtxt(dpath+'fig6a-fft-hanning.dat')
#oms = np.loadtxt(dpath+'upto70.dat')

res = oms[:, 1]*np.exp(oms[:, 0]**2/16)
plt.plot(oms[:500, 0], res[:500])