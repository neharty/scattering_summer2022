import numpy as np
import matplotlib.pyplot as plt
from tds_jit import scattering as scjit
from Ncyl.tds import scattering, cyl
from time import process_time
from utils import example_setups

np.random.seed(666)
cyls = example_setups.random_hexagon(5, 'd', 1, 3)

t = process_time()
scattering(cyls).scattering_coeffs()
print(process_time() - t)

t = process_time()
scjit(cyls).scattering_coeffs()
print(process_time() - t)


