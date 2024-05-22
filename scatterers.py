import numpy as np
from abc import ABC, abstractmethod


class scatterer(ABC):
    '''
        abstract scatterer class
    '''
    def __init__(self, bc, pos, radius, rotation, sum_index = None, impedance=None, uid = None, maxidx = np.inf):
        self.label = 0
        self.pos = np.array(pos) # center position of the scatterer
        self.radius = radius #radius of the circle enclosing the scatterer
        self.bc = bc #boundary condition
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
        self.name = 'cyl'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, radius, 0, sum_index, impedance, self.uid, self.maxidx)
    
    def bdy(self, t):
        return self.radius*(np.array([np.cos(t), np.sin(t)]))
    
    def dbdy(self, t):
        return self.radius*(np.array([-np.sin(t), np.cos(t)]))
    
class starfish(scatterer):
    '''
        starfish shape
    '''
    def __init__(self, bc, pos, radius, rotation, sum_index = None, impedance=None):
        self.label = 0
        self.name = 'starfish'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, radius, rotation, sum_index, impedance, self.uid, self.maxidx)
        
    def bdy(self, t):
        r = lambda x : 9/10 - (1/9)*np.cos(5*x) #has a max distance of 1
        return self.radius*r(t)*np.array([np.cos(t), np.sin(t)])
    
    def dbdy(self, t):
        r = lambda x : 9/10 - (1/9)*np.cos(5*x)
        dr = lambda x : (5/9)*np.sin(5*x) 
        return self.radius*(r(t)*np.array([-np.sin(t), np.cos(t)])+dr(t)*np.array([np.cos(t), np.sin(t)]))
