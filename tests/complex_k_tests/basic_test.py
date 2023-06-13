import numpy as np

from Ncyl.tds import scattering
from utils import example_setups

cyls = example_setups.random_hexagon(4, 'd', 1, 3)
sc = scattering(cyls, freq=1j)
sc.scattering_coeffs()
print('solved')
