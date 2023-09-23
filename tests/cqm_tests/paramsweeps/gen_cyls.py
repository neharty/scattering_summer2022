import pickle
from Ncyl.cqm import cyl
from utils import example_setups


outpfile = './cyls.p'
cyls = example_setups.uniform_grid_NxN(3, 'd', 1, 3)
pickle.dump(cyls, open(outpfile, 'wb'))
