import numpy as np
from Ncyl.Ncyl import scattering, cyl
import matplotlib.pyplot as plt

M_sums = np.arange(1, 26)
conds = np.zeros(len(M_sums))

fig, ax = plt.subplots()

for i in range(len(M_sums)):
    sc = scattering([cyl(0, 'n', pos = np.array([0, -1]), radius = 0.3), cyl(1, 'd', pos = np.array([0, 1]), radius = 0.3)], M_sum = M_sums[i])
    matrix = sc.make_scattering_blocks()
    if not np.isnan(np.linalg.cond(matrix)):
        conds[i] = np.linalg.cond(matrix)


ax.semilogy(M_sums, conds, '.-')
ax.set_xlabel('sum index')
ax.set_ylabel('cond num')
fig.suptitle('radius = 0.3, dist = 2')
plt.savefig('quick_cond_num_study.png')
plt.show()



