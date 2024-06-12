import numpy as np
from abc import ABC, abstractmethod

class scatterer(ABC):
    '''
        abstract scatterer class
    '''
    def __init__(self, bc, pos, radius, rotation, sum_index = None, impedance=None, uid = None, maxidx = np.inf):
        self.label = 0
        self.pos = np.array(pos) # center position of the scatterer
        self.radius = float(radius) #radius of the circle enclosing the scatterer
        self.rotation = rotation #rotation angle of the scatterer
        self.bc = bc #boundary condition
        self.impedance = impedance #impedance is a function handle for \eta
        
        if self.bc == 'i' and impedance == None:
            print(impedance)
            raise NotImplementedError("Impedance is not set")
        
        self.sum_index = sum_index
        self.uid = uid #used for disk writes
        self.maxidx = maxidx #used for z-transform
    
    @abstractmethod
    def r(self, t):
        pass
    
    @abstractmethod
    def dr(self, t):
        pass
    
    def bdy(self, t):
        return self.radius*self.r(t)*np.array([np.cos(t), np.sin(t)])
    
    def dbdy(self, t):
        return self.radius*(self.r(t)*np.array([-np.sin(t), np.cos(t)])+self.dr(t)*np.array([np.cos(t), np.sin(t)]))

class cyl(scatterer):
    '''
        cylinder, or circle, in the plane
    '''
    def __init__(self, bc, pos, radius, rotation=0., sum_index = None, impedance = None):
        self.label = 0
        self.name = 'cyl'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, radius, rotation, sum_index, impedance, self.uid, self.maxidx)
    
    def r(self, t):
        return 1.0
    
    def dr(self, t):
        return 0.0
 
class fat_starfish(scatterer):
    def __init__(self, bc, pos, radius, rotation, sum_index = None, impedance=None):
        self.label = 0
        self.name = 'fat_starfish'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, radius, rotation, sum_index, impedance, self.uid, self.maxidx)
    
    def r(self, t):
        return 9/10 - (1/10)*np.cos(5*(t-self.rotation)) #has a max distance of 1
    
    def dr(self, t):
        return np.sin(5*(t-self.rotation))/2
 
class starfish(scatterer):
    def __init__(self, bc, pos, radius, rotation, sum_index = None, impedance=None):
        self.label = 0
        self.name = 'starfish'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, radius, rotation, sum_index, impedance, self.uid, self.maxidx)
        
    def r(self, t):
        return 4/5 - (1/5)*np.cos(5*(t-self.rotation)) #has a max distance of 1
    
    def dr(self, t):
        return np.sin(5*(t-self.rotation)) 

class bean(scatterer):
    def __init__(self, bc, pos, radius, rotation, sum_index = None, impedance=None):
        self.label = 0
        self.name = 'bean'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, radius, rotation, sum_index, impedance, self.uid, self.maxidx)
        
    def r(self, t):
        #might change these coefficients to 4/5 and 1/5
        return (np.cos(t-self.rotation))**3 + (np.sin(t-self.rotation))**3
    
    def dr(self, t):
        return 3*(np.cos(t-self.rotation))**2*(-np.sin(t-self.rotation)) + 3*(np.sin(t-self.rotation))**2*np.cos(t-self.rotation)
    
    def bdy(self, t):
        tmp = self.radius*self.r(t)*np.array([np.cos(t), np.sin(t)])
        tmp2 = tmp.T - np.array([0.3, 0.3])
        return tmp2.T
    
    def dbdy(self, t):
        tmp = self.radius*self.r(t)*np.array([np.cos(t), np.sin(t)])
        tmp2 = tmp.T - np.array([0.3, 0.3])
        return self.radius*(self.r(t)*np.array([-np.sin(t), np.cos(t)]))+tmp2.T
    
class leaf(scatterer):
    def __init__(self, bc, pos, radius, rotation, sum_index = None, impedance=None):
        self.label = 0
        self.name = 'leaf'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, radius, rotation, sum_index, impedance, self.uid, self.maxidx)
        
    def r(self, t):
        #might change these coefficients to 4/5 and 1/5
        return (1+0.9*np.cos(t - self.rotation))*(1+0.1*np.cos(20*(t-self.rotation)))*(0.9+0.1*np.cos(200*(t-self.rotation)))*(1+np.sin(t-self.rotation))/4.18
    
    def dr(self, t):
        return (0.009*np.cos(t-self.rotation)*(np.cos(8*(t-self.rotation))+1.11111)*(np.cos(24*(t-self.rotation))+10)*(np.cos(200*(t-self.rotation))+9)\
                -0.072*(np.sin(t-self.rotation)+1)*np.sin(8*(t-self.rotation))*(np.cos(24*(t-self.rotation))+10)*(np.cos(200*(t-self.rotation))+9)\
                -0.216*(np.sin(t-self.rotation)+1)*np.sin(24*(t-self.rotation))*(np.cos(8*(t-self.rotation))+1.11111)*(np.cos(200*(t-self.rotation))+9)\
                -1.8*(np.sin(t-self.rotation)+1)*(np.cos(8*(t-self.rotation))+1.11111)*(np.cos(24*(t-self.rotation))+10))/4.18