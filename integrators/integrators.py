import numpy as np
from abc import ABC, abstractmethod

# implementation of timestepping integrators using butcher tables

class integrator(ABC):
    '''
        Template class for butcher tables
    '''
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c


class IRK3(integrator):
    def __init__(self):
        Amat = np.array([[5/12, -1/12], 
                         [3/4, 1/4]])
        bmat = np.array([3/4, 1/4])
        cmat = np.array([1/3, 1])
        self.order = 3
        super().__init__(Amat, bmat, cmat)
        
class IRK5(integrator):
    def __init__(self):
        A = np.array([[11/45 - 7*np.sqrt(6)/360, 37/225 - 169*np.sqrt(6)/1800, -2/225 + np.sqrt(6)/75],
                      [37/225 + 169*np.sqrt(6)/1800, 11/45 + 7*np.sqrt(6)/360, -2/225 - np.sqrt(6)/75],
                      [4/9 - np.sqrt(6)/36, 4/9 + np.sqrt(6)/36, 1/9]])
        b = np.array([4/9 - np.sqrt(6)/36, 4/9 + np.sqrt(6)/36, 1/9])
        c = np.array([2/5 - np.sqrt(6)/10, 2/5 + np.sqrt(6)/10, 1])
        self.order=5
        super().__init__(A, b, c)