import numpy as np
import multiprocessing as mp
import 
import os
#minimal parallel example

class Overlord():
    def __init__(self, n, m):
        self.rhsmat = np.random.rand(n, m)
        opath='/home/nick/scattering/tmp/test/'
        os.makedirs('/home/nick/scattering/tmp/test/', exist_ok = True)
        np.savetxt(opath+'rhs.dat')
    
    @staticmethod
    def comp(idx):
        M = Matrix(self.rhsmat[:, idx])
        return M.solve()
   
    def foo(self):    

        
        return res
    
