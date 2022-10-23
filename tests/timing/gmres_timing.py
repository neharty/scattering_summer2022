import numpy as np
import matplotlib.pyplot as plt
from time import process_time()

from Ncyl.Ncyl_gmres import scattering, cyl
from utils import example_setups

N = 10
cyls = example_setups.triangular_lattice(N, N, ['d' for i in range(N)], 1, 10, 10)
sc = scattering(cyls, 
