import numpy as np
import multiprocessing as mp
#minimal parallel example

class Matrix():
    def __init__(self, rhs):
        self.rhs = rhs
    
    def make_matrix(self):
        return np.random.rand(len(self.rhs), 
                                  len(self.rhs))
        
    def solve(self):
        return np.linalg.solve(self.make_matrix, 
                               self.rhs)
    
class Overlord():
    def __init__(self, n, m):
        self.rhsmat = np.random.rand(n, m)
    
    @staticmethod
    def comp(idx):
        M = Matrix(self.rhsmat[:, idx])
        return M.solve()
   
    def foo(self):    
        pool = mp.Pool(processes=mp.cpu_count())
        res = pool.map(self.comp, range(self.rhsmat.shape[0]))
        pool.close()
        pool.join()
        
        return res
    
