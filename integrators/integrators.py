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
        super().__init__(Amat, bmat, cmat)