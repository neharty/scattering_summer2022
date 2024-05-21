import numpy as np
from abc import ABC, abstractmethod


class scatterer(ABC):
    '''
        abstract scatterer class
    '''
    def __init__(self, bc, pos, radius, sum_index = None, impedance=None, uid = None, maxidx = np.inf):
        self.label = 0
        self.pos = np.array(pos) # center position of the scatterer
        self.radius = radius
        self.bc = bc
        self.impedance = impedance #impedance is a function handle for \eta
        
        if self.bc == 'i' and impedance == None:
            print(impedance)
            raise NotImplementedError("Impedance is not set")
        
        self.sum_index = sum_index
        self.uid = uid #used for disk writes
        self.maxidx = maxidx #used for z-transform
        
    @abstractmethod
    def bdy(self, t):
        pass
    
    @abstractmethod
    def dbdy(self, t):
        pass

class cyl(scatterer):
    '''
        cylinder, or circle, in the plane
    '''
    def __init__(self, bc, pos, radius, sum_index = None, impedance = None):
        self.label = 0
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, radius, sum_index, impedance, self.uid, self.maxidx)
    
    def bdy(self, t):
        return self.radius*(np.array([np.cos(t), np.sin(t)]))
    
    def dbdy(self, t):
        return self.radius*(np.array([-np.sin(t), np.cos(t)]))
    
class starfish(scatterer):
    '''
        starfish shape
    '''
    def __init__(self, bc, pos, radius, sum_index = None, impedance=None):
        self.label = 0
        self.radius = 1
        super().__init__(bc, pos, radius, sum_index, impedance, self.uid, self.maxidx)
        
    def bdy(self, t):
        pass
    
    def dbdy(self, t):
        pass
