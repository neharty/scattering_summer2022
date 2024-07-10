import numpy as np
from abc import ABC, abstractmethod


def scat_select(scrname, bc, pos, sc_scale, rotation, sum_num=None, impedance=None, rad_scale=1):
     if scrname == 'cyl':
         return cyl(bc, pos, sc_scale, 0, sum_num, impedance, rad_scale)
     elif scrname == 'starfish':
         return starfish(bc,  pos, sc_scale, rotation, sum_num, impedance, rad_scale)
     elif scrname == 'fat_starfish':
         return fat_starfish(bc, pos, sc_scale, rotation, sum_num, impedance, rad_scale)
     elif scrname == 'triad':
         return triad(bc, pos, sc_scale, rotation, sum_num, impedance, rad_scale)
     elif scrname == 'smooth_hexagon':
         return smooth_hexagon(bc, pos, sc_scale, rotation, sum_num, impedance, rad_scale)
     elif scrname == 'teardrop':
         return teardrop(bc, pos, sc_scale, rotation, sum_num, impedance, rad_scale)
     elif scrname == 'strip':
         return strip(bc, pos, sc_scale, rotation, sum_num, impedance, rad_scale)
     elif scrname == 'boomerang':
         return boomerang(bc, pos, sc_scale, rotation, sum_num, impedance, rad_scale)
     elif scrname == 'bean':
         return bean(bc, pos, sc_scale, rotation, sum_num, impedance, rad_scale)

class scatterer(ABC):
    '''
        abstract scatterer class
    '''
    def __init__(self, bc, pos, sc_scale, rotation, sum_index = None, impedance=None, uid = None, maxidx = np.inf, rad_scale=1):
        self.label = 0
        self.pos = np.array(pos) # center position of the scatterer
        self.sc_scale = float(sc_scale)
        self.radius = float(rad_scale*sc_scale) #radius of the circle enclosing the scatterer
        self.rotation = rotation #rotation angle of the scatterer
        self.bc = bc #boundary condition
        self.impedance = impedance #impedance is a function handle for \eta
        self.offset_t = False
        self.Rmat = np.array([[np.cos(self.rotation), -np.sin(self.rotation)], 
                              [np.sin(self.rotation), np.cos(self.rotation)]])
        
        if self.bc == 'i' and impedance == None:
            print(impedance)
            raise NotImplementedError("Impedance is not set")
        
        self.sum_index = sum_index
        self.uid = uid #used for disk writes
        self.maxidx = maxidx #used for z-transform
        
        self.sigma = 4 

    def v(self, s, ti, tii):
        return (1/self.sigma - 0.5)*((ti + tii - 2*s)/(tii - ti))**3 + (2*s - ti - tii)/(self.sigma*(tii - ti)) + 0.5
    
    def dv(self, s, ti, tii):
        return 3*(1/self.sigma - 0.5)*((ti + tii - 2*s)/(tii - ti))**2*(-2/(tii - ti)) + 2/(self.sigma*(tii - ti))

    def w(self, s, ti, tii):
        return (tii*self.v(s, ti, tii)**self.sigma + ti*(1 - self.v(s, ti, tii))**self.sigma)/(self.v(s, ti, tii)**self.sigma + (1-self.v(s, ti, tii))**self.sigma) 
    
    def dw(self, s, ti, tii):
        return ((self.v(s, ti, tii)**self.sigma + (1-self.v(s, ti, tii))**self.sigma) \
                *(tii*self.sigma*self.v(s, ti, tii)**(self.sigma-1)*self.dv(s, ti, tii) - ti*self.sigma*(1-self.v(s, ti, tii))**(self.sigma-1)*self.dv(s, ti, tii)) \
                    + (self.sigma*self.v(s, ti, tii)**(self.sigma-1)*self.dv(s, ti, tii) - self.sigma*(1 - self.v(s, ti, tii))**(self.sigma-1)*self.dv(s, ti, tii)) \
                        *(tii*self.v(s, ti, tii)**self.sigma + ti*(1 - self.v(s, ti, tii))**self.sigma)) \
            /(self.v(s, ti, tii)**self.sigma + (1-self.v(s, ti, tii))**self.sigma)**2
    
    @abstractmethod
    def r(self, t):
        pass
    
    @abstractmethod
    def dr(self, t):
        pass
    
    def bdy(self, t):
        return self.sc_scale*self.r(t)*np.array([np.cos(t), np.sin(t)])
    
    def dbdy(self, t):
        return self.sc_scale*(self.r(t)*np.array([-np.sin(t), np.cos(t)])+self.dr(t)*np.array([np.cos(t), np.sin(t)]))

class cyl(scatterer):
    '''
        cylinder, or circle, in the plane
    '''
    def __init__(self, bc, pos, sc_scale, rotation=0., sum_index = None, impedance = None, rad_scale=1):
        self.label = 0
        self.name = 'cyl'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, sc_scale, rotation, sum_index, impedance, self.uid, self.maxidx, rad_scale)
    
    def r(self, t):
        return 1.0
    
    def dr(self, t):
        return 0.0
 
class fat_starfish(scatterer):
    def __init__(self, bc, pos, sc_scale, rotation, sum_index = None, impedance=None, rad_scale=1):
        self.label = 0
        self.name = 'fat_starfish'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, sc_scale, rotation, sum_index, impedance, self.uid, self.maxidx, rad_scale)
    
    def r(self, t):
        return 9/10 - (1/10)*np.cos(5*(t-self.rotation)) 
    
    def dr(self, t):
        return np.sin(5*(t-self.rotation))/2
 
class triad(scatterer):
    def __init__(self, bc, pos, sc_scale, rotation, sum_index = None, impedance=None, rad_scale=1):
        self.label = 0
        self.name = 'triad'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, sc_scale, rotation, sum_index, impedance, self.uid, self.maxidx, rad_scale)
        
    def r(self, t):
        return 4/5 - (1/5)*np.cos(3*(t-self.rotation)) 
    
    def dr(self, t):
        return 3*np.sin(3*(t-self.rotation))/5
    
class starfish(scatterer):
    def __init__(self, bc, pos, sc_scale, rotation, sum_index = None, impedance=None, rad_scale=1):
        self.label = 0
        self.name = 'starfish'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, sc_scale, rotation, sum_index, impedance, self.uid, self.maxidx, rad_scale)
        
    def r(self, t):
        return 4/5 - (1/5)*np.cos(5*(t-self.rotation)) 
    
    def dr(self, t):
        return np.sin(5*(t-self.rotation)) 
    
class smooth_hexagon(scatterer):
    def __init__(self, bc, pos, sc_scale, rotation, sum_index = None, impedance=None, rad_scale=1):
        self.label = 0
        self.name = 'smooth_hexagon'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, sc_scale, rotation, sum_index, impedance, self.uid, self.maxidx, rad_scale)
        
    def r(self, t):
        return 4/5 - (1/5)*np.cos(6*(t-self.rotation)) 
    
    def dr(self, t):
        return 6*np.sin(6*(t-self.rotation))/5

class teardrop(scatterer):
    def __init__(self, bc, pos, sc_scale, rotation, sum_index = None, impedance=None, rad_scale=1):
        self.label = 0
        self.name = 'teardrop'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, sc_scale, rotation, sum_index, impedance, self.uid, self.maxidx, rad_scale)
        a = 0.5
        self.b = np.tan(a*np.pi/2)/2
        self.offset_t = True

    def r(self, t):
        return 0
    
    def dr(self, t):
        return 0
    
    def bdy(self, t):
        return self.sc_scale*self.Rmat @ np.array([np.sin(self.w(t, 0, 2*np.pi)/2), -self.b*np.sin(self.w(t, 0, 2*np.pi))])
        
    def dbdy(self, t):
        return self.sc_scale*self.Rmat @ np.array([np.cos(self.w(t, 0, 2*np.pi)/2)/2, -self.b*np.cos((self.w(t, 0, 2*np.pi)))])
        
class boomerang(scatterer):
    def __init__(self, bc, pos, sc_scale, rotation, sum_index = None, impedance=None, rad_scale=1):
        self.label = 0
        self.name = 'boomerang'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, sc_scale, rotation, sum_index, impedance, self.uid, self.maxidx, rad_scale)
        self.offset_t = True
        
    def r(self, t):
        return 0
    
    def dr(self, t):
        return 0
    
    def bdy(self, t):
        return self.sc_scale*self.Rmat @ np.array([-(2/3)*np.sin(3*(self.w(t, 0, 2*np.pi))/2), -np.sin(self.w(t, 0, 2*np.pi))])/(np.sqrt(21)/4)
        
    def dbdy(self, t):
        return self.sc_scale*self.Rmat @ np.array([-np.cos(3*(self.w(t, 0, 2*np.pi))/2), -np.cos((self.w(t, 0, 2*np.pi)))])/(np.sqrt(21)/4)

class strip(scatterer):
    def __init__(self, bc, pos, sc_scale, rotation, sum_index = None, impedance=None, rad_scale=1):
        self.label = 0
        self.name = 'strip'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, sc_scale, rotation, sum_index, impedance, self.uid, self.maxidx, rad_scale)
        self.offset_t = True
        
    def r(self, t):
        return 0
    
    def dr(self, t):
        return 0
    
    def bdy(self, t):
        return self.sc_scale*self.Rmat @ np.array([self.w(t, 0, 2*np.pi)/np.pi - 1, 0*t])
        
    def dbdy(self, t):
        return self.sc_scale*self.Rmat @ np.array([1/np.pi, 0*t])

class vstrip(scatterer):
    pass

class bean(scatterer):
    def __init__(self, bc, pos, radius, rotation, sum_index = None, impedance=None, rad_scale=1):
        self.label = 0
        self.name = 'bean'
        self.uid = None
        self.maxidx = np.inf
        super().__init__(bc, pos, radius, rotation, sum_index, impedance, self.uid, self.maxidx, rad_scale)
        self.offset_t = True
        
    def r(self, t):
        return (np.cos(t-self.rotation))**3 + (np.sin(t-self.rotation))**3
    
    def dr(self, t):
        return 3*(np.cos(t-self.rotation))**2*(-np.sin(t-self.rotation)) + 3*(np.sin(t-self.rotation))**2*np.cos(t-self.rotation)
    
# class leaf(scatterer):
#     def __init__(self, bc, pos, radius, rotation, sum_index = None, impedance=None, rad_scale=1):
#         self.label = 0
#         self.name = 'leaf'
#         self.uid = None
#         self.maxidx = np.inf
#         super().__init__(bc, pos, radius, rotation, sum_index, impedance, self.uid, self.maxidx, rad_scale)
        
#     def r(self, t):
#         #might change these coefficients to 4/5 and 1/5
#         return (1+0.9*np.cos(t - self.rotation))*(1+0.1*np.cos(20*(t-self.rotation)))*(0.9+0.1*np.cos(200*(t-self.rotation)))*(1+np.sin(t-self.rotation))/4.18
    
#     def dr(self, t):
#         return (0.009*np.cos(t-self.rotation)*(np.cos(8*(t-self.rotation))+1.11111)*(np.cos(24*(t-self.rotation))+10)*(np.cos(200*(t-self.rotation))+9)\
#                 -0.072*(np.sin(t-self.rotation)+1)*np.sin(8*(t-self.rotation))*(np.cos(24*(t-self.rotation))+10)*(np.cos(200*(t-self.rotation))+9)\
#                 -0.216*(np.sin(t-self.rotation)+1)*np.sin(24*(t-self.rotation))*(np.cos(8*(t-self.rotation))+1.11111)*(np.cos(200*(t-self.rotation))+9)\
#                 -1.8*(np.sin(t-self.rotation)+1)*(np.cos(8*(t-self.rotation))+1.11111)*(np.cos(24*(t-self.rotation))+10))/4.18