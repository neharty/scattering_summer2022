import os
import numpy as np
import multiprocessing as mp

dpath = None

rhs=
    
def make_matrix():
    return np.random.rand(len(rhs), 
                              len(rhs))
    
def solve():
    return np.linalg.solve(make_matrix, 
                           rhs)


pool = mp.Pool(processes=mp.cpu_count())
res = pool.map(solve(), range(self.rhsmat.shape[0]))
pool.close()
pool.join()