import numpy as np
from abc import ABC, abstractmethod

class waveform(ABC):
    '''
        preconditioner template class for
        scattering-based preconditioners
    '''
    def __init__(self, inc_angle = np.pi, time_shift = 0):
        self.inc_angle = inc_angle
        self.time_shift = time_shift
        
    def inparg(self, x, y, t):
        return t - (x*np.cos(self.inc_angle) + y*np.sin(self.inc_angle)) - self.time_shift
        
    def d_inparg(self, x, y, t):
        return -np.array([np.cos(self.inc_angle), np.sin(self.inc_angle)])
    
    @abstractmethod
    def profile(self, z):
        pass

    def td_wave(self, x, y, t):
        return self.profile(self.inparg(x, y, t))
    
    @abstractmethod
    def td_deriv(self, x, y, t, n):
        pass

class gaussian(waveform):
    def __init__(
                 self, 
                 inc_angle = np.pi, 
                 time_shift=0, 
                 sigma = 0.2, 
                 cutoff=1e-8,
                 amp=1
                ):
        super().__init__(time_shift=time_shift, inc_angle=inc_angle)
        self.cutoff = 1e-8 #used to truncate signal
        self.sigma = sigma
        self.amp = amp
    
    def box(self, x, l):
        mask1 = (x <= l)
        mask2 = (x >= -l)
        return mask1*mask2

    def profile(self, z):
        #waveform profile, centered at the origin
        s = self.sigma
        tmp = self.amp*np.exp(-(z**2)/s)

        #cutoff width
        l = np.sqrt(-s*np.log(self.cutoff))+1
        return self.box(z, l)*tmp  
    
    def td_deriv(self, x, y, t, n):
        v= (-self.inparg(x, y, t)*2/self.sigma)*self.td_wave(x, y, t)
        dg = np.zeros(len(x))
        for i in range(len(x)):
            dg[i] = self.d_inparg(x, y, t).dot(n[i])
        dg *= v
        return dg
        
class gaussian_deriv(waveform):
    def __init__(
                 self, 
                 inc_angle = np.pi, 
                 time_shift=0, 
                 sigma = 0.2,
                 amp = 1, 
                 cutoff=1e-8
                ):
        super().__init__(time_shift=time_shift, inc_angle=inc_angle)
        self.cutoff = 1e-8 #used to truncate signal
        self.sigma = sigma
        self.amp = amp
    
    def box(self, x, l):
        mask1 = (x <= l)
        mask2 = (x >= -l)
        return mask1*mask2

    def profile(self, z):
        #waveform profile, centered at the origin
        s = self.sigma
        tmp = self.amp*np.sqrt(2/s)*z*np.exp(0.5-(z**2)/s)

        #approximate cutoff width
        l = np.sqrt(s*np.log(np.exp(0.5)*np.sqrt(2/s)/self.cutoff))+1
        return self.box(z, l)*tmp
    
    def td_deriv(self, x, y, t, n):
        v = self.amp*np.sqrt(2/self.sigma)*np.exp(0.5-(self.inparg(x, y, t)**2)/self.sigma)*(1 - 2*(self.inparg(x, y, t))**2/(self.sigma))
        dg = np.zeros(len(x))
        for i in range(len(x)):
            dg[i] = self.d_inparg(x, y, t).dot(n[i])
        dg *= v
        return dg

class kachanovska(waveform):
    #wave from kachanovska 
    def __init__(self, inc_angle = np.pi/2, time_shift=2):
        super().__init__(inc_angle, time_shift)
        
    def profile(self, z):
        return np.exp(-20*z**2)*np.sin(30*z)
    
    def td_deriv(self, x, y, t, n):
        dv = None #TODO: replace with derivative
        dg = np.zeros(len(x))
        for i in range(len(x)):
            dg[i] = self.d_inparg(x, y, t).dot(n[i])
        dg *= dv
        return dg

class sine_pulse(waveform):
    def __init__(
                 self, 
                 inc_angle = np.pi, 
                 time_shift=0, 
                 alpha = 0.5, 
                 width = 20, 
                 cycles = 1,
                ):
        super().__init__(time_shift=time_shift, inc_angle=inc_angle)
        self.alpha = alpha
        self.width = width
        self.sf = cycles*np.pi*2/(width/4)
    
    def box(self, x):
        mask1 = (x <= self.width/4)
        mask2 = (x >= -self.width/4)
        return mask1*mask2

    def profile(self, z):
        #waveform profile, centered at the origin
        tmp = np.sin(self.sf*z)*np.exp(-self.alpha*np.abs(z))
        return self.box(z)*tmp
    
    def td_deriv(self, x, y, t, n):
        pass

class sinewave(waveform):
    def __init__(
                 self, 
                 inc_angle = np.pi, 
                 time_shift=0, 
                 width = 4*np.pi, 
                 cycles = 1,
                ):
        super().__init__(time_shift=time_shift, inc_angle=inc_angle)
        self.width = width
        self.sf = cycles*np.pi/(width/4)
    
    def box(self, x):
        mask1 = (x <= self.width/4)
        mask2 = (x >= -self.width/4)
        return mask1*mask2

    def profile(self, z):
        #waveform profile, centered at the origin
        tmp = np.sin(self.sf*z)
        return self.box(z)*tmp
    
    def td_deriv(self, x, y, t, n):
        pass
    
class russian(waveform):
    def __init__(
                 self, 
                 inc_angle = np.pi, 
                 time_shift=0, 
                 alpha= 1,
                 beta= 1,
                 cycles = 1,
                ):
        super().__init__(time_shift=time_shift, inc_angle=inc_angle)
        self.alpha = alpha
        self.beta = beta
    
    def profile(self, z):
        #waveform profile, centered at the origin
        mask = (z >= 0)
        tmp = mask*self.alpha*(z**2)*np.exp(-self.beta*z)
        return tmp

    def td_wave(self, x, y, t):
        return self.profile(self.inparg(x, y, t))
    
    def td_deriv(self, x, y, t, n):
        pass


'''
class bump():
    pass

    def bump(x, y, t):
        z = x + t - shift
        mask = z > -1 & z < 1
        fn = np.exp(-1/(1 - (z*mask)**2))
        return mask*fn
'''
