import numpy as np
import sys
from Ncyl.Ncyl_gmres import scattering, cyl
from utils import example_setups

np.set_printoptions(threshold=sys.maxsize)

N = 5
cyls = example_setups.uniform_grid_NxN(N, ['d' for i in range(N)], 1, 5)
eps = 1e-12
sc = scattering(cyls, sum_tol=eps, gmres_tol=eps)
sc_precond = scattering(cyls, precond=True, precond_dist=5, sum_tol=eps, gmres_tol=eps)

sc.scattering_coeffs()
sc_precond.scattering_coeffs()

print(sc.get_gmres_iter(), sc_precond.get_gmres_iter())
