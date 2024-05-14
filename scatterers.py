import numpy as np
from abc import ABC, abstractmethod


class scatterer(ABC):
    '''
        abstract scatterer class
    '''
    def __init__(self, bc, pos, sum_index = None, impedance=None):
        self.label = 0
        self.pos = np.array(pos) # center position of the scatterer
        self.radius = radius # radius of the circle enclosing the scatterer
        self.bc = bc
        self.impedance = impedance #impedance is a function handle for \eta
        
        if self.bc == 'i' and impedance == None:
            print(impedance)
            raise NotImplementedError("Impedance is not set")
        
        self.sum_index = sum_index
        self.uid = None #used for disk writes
        self.maxidx = np.inf #used for z-transform

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
    
    
        
    def get_label(self):
        return self.label
    
