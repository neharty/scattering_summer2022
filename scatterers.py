import numpy as np
from abc import ABC, abstractmethod


class scatterer(ABC):
    '''
        abstract scatterer class
    '''
    def __init__(self, bc, pos, sum_index = None, impedance=None):
        self.label = 0
        self.pos = np.array(pos) # center position of the scatterer
        self.bc = bc
        self.impedance = impedance #impedance is a function handle for \eta
        
        if self.bc == 'i' and impedance == None:
            print(impedance)
            raise NotImplementedError("Impedance is not set")
        
        self.sum_index = sum_index
        self.uid = None #used for disk writes
        self.maxidx = np.inf #used for z-transform
        
    @abstractmethod
    def curve(self, t):
        pass
    
    @abstractmethod
    def dcurve(self, t):
        pass

class cyl(scatterer):
    '''
    Class to build 1 cylinder. Cylinders are identified by a
    number label, set by the user or by the scattering class.
    '''

    def __init__(self, bc, pos, radius, sum_index = None, impedance = None):
        self.label = 0
        self.pos = np.array(pos)
        self.radius = radius
        self.bc = bc
        self.impedance = impedance #impedance is a function handle for \eta
        if self.bc == 'i' and impedance == None:
            print(impedance)
            raise NotImplementedError("Impedance is not set")
        self.sum_index = sum_index
        self.uid = None #used for disk writes
        self.maxidx = np.inf #used for z-transform
    
    def curve(self, t):
        return self.radius*(np.array([np.cos(t), np.sin(t)])) + self.pos
    
    def dcurve(self, t):
        return self.radius*(np.array([-np.sin(t), np.cos(t)])) + self.pos
    
class starfish(scatterer):
    '''
        abstract scatterer class
    '''
    def __init__(self, bc, pos, sum_index = None, impedance=None):
        self.label = 0
        self.pos = np.array(pos) # center position of the scatterer
        self.bc = bc
        self.impedance = impedance #impedance is a function handle for \eta
        
        if self.bc == 'i' and impedance == None:
            print(impedance)
            raise NotImplementedError("Impedance is not set")
        
        self.sum_index = sum_index
        self.uid = None #used for disk writes
        self.maxidx = np.inf #used for z-transform
        
    def curve(self, t):
        pass
    
    def dcurve(self, t):
        pass
