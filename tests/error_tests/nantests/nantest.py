import numpy as np
import sys
sys.path.append('../../Ncyl')
import Ncyl
from Ncyl import scattering, cyl
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1, yv
from Ncyl import S

M_sums = np.arange(100, 160, 10)
conds = np.zeros(len(M_sums))

fig, ax = plt.subplots()
for i in range(len(M_sums)):
    sc = scattering([cyl(0, 'd', pos = np.array([0, -1]), radius = 0.3), cyl(1, 'd', pos = np.array([0, 1]), radius = 0.3)], M_sum = M_sums[i])
    matrix = sc.make_scattering_blocks()
    if not np.isnan(np.linalg.cond(matrix)):
        conds[i] = np.linalg.cond(matrix)
    else:
        M = M_sums[i]
        print(M)
        print(2*M+1)
        print(np.argwhere(np.isnan(matrix)))
        print(jv(-2*M, Ncyl.k*0.3), S(-M, M, np.array([0, 2])), jv(-M, Ncyl.k*0.3)* S(-M, M, np.array([0, 2])))
        print(hankel1(-2*M, Ncyl.k*0.3)) #it looks like the hankel function is causing nan values
        

'''
ax.semilogy(M_sums, conds, '.-')
ax.set_xlabel('sum index')
ax.set_ylabel('cond num')
fig.suptitle('radius = 0.3, dist = 2')
plt.savefig('quick_cond_num_study.png')
plt.show()
'''


