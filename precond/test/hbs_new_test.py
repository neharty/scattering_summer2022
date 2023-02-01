import numpy as np
from Ncyl.tds import scattering, cyl
from utils import example_setups
from precond import preconditioner

np.random.seed(666)

dist = 4
cyls = example_setups.random_pos(-10, 10, -10, 10, 15, 'd', 1, dist)
sc = scattering(cyls)
A_inv = preconditioner.hbs(sc).inv_mat

