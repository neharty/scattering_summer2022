import numpy as np
#import matplotlib.pyplot as plt
from time import process_time

from Ncyl.tds import scattering
from utils import example_setups

np.random.seed(12345)
cyls = example_setups.random_hexagon(5, 'd', 1, 3)

t = process_time()
sc = scattering(cyls)
init_time = process_time() - t

t = process_time()
sc.make_root_matrix()
rootmat_time = process_time() - t

t = process_time()
sc.scattering_coeffs()
gmres_time = process_time() - t


print('init time:', init_time)
print('rootmat comp time:', rootmat_time)
print('gmres time:', gmres_time)

