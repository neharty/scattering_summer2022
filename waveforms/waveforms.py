import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.special import hankel1, h1vp
from abc import ABC, abstractmethod

class waveform(ABC):
    '''
        template class for waveforms
    '''
    def __init__(
                 self, 
                 inc_angle = 0, 
                 time_shift = 0, 
                 r0 = np.array([0, 0]), 
                 src_type = 'planewave'
                 ):
        self.inc_angle = inc_angle
        self.phi = np.pi+self.inc_angle
        self.time_shift = time_shift
        self.src_type = src_type
        self.r0 = np.array(r0)
        self.times = None #used for point source times
        
        
    def inparg(self, x, y, t):
        return t - (x*np.cos(self.phi) + y*np.sin(self.phi)) - self.time_shift
        
    def d_inparg(self, x, y, t):
        return -np.array([np.cos(self.phi), np.sin(self.phi)])
    
    @abstractmethod
    def profile(self, z):
        pass
    
    def ptsrc_init(self, t, lammat):
        self.times = t
        self.coeffs = fft(lammat*self.profile(self.times - self.time_shift))
    
    def td_wave(self, x, y, t):
        return self.profile(self.inparg(x, y, t))

    @abstractmethod
    def td_deriv(self, x, y, t, n):
        pass

class gaussian(waveform):
    def __init__(
                 self, 
                 inc_angle = 0, 
                 time_shift=0, 
                 r0 = [0,0],
                 src_type = 'planewave',
                 sigma = 0.2, 
                 cutoff=1e-10,
                 amp=1
                ):
        super().__init__(time_shift=time_shift, 
                         inc_angle=inc_angle, 
                         r0=r0,
                         src_type=src_type)
        self.cutoff = cutoff #used to truncate signal
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
                 inc_angle = 0, 
                 time_shift=0,
                 r0 = [0, 0],
                 src_type='planewave',
                 sigma = 0.2,
                 amp = 1, 
                 cutoff=1e-10
                ):
        super().__init__(time_shift=time_shift, 
                         inc_angle=inc_angle,
                         r0 = r0,
                         src_type=src_type
                         )
        self.cutoff = cutoff #used to truncate signal
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
    def __init__(self, inc_angle = -np.pi/2, time_shift=2):
        super().__init__(inc_angle, time_shift)
        
    def profile(self, z):
        return np.exp(-20*z**2)*np.sin(30*z)
    
    def td_deriv(self, x, y, t, n):
        dv = np.exp(-20*self.inparg(x, y, t)**2)*(30*np.cos(30*self.inparg(x, y, t)) - 40*self.inparg(x, y, t)*np.sin(30*self.inparg(x, y, t)))
        dg = np.zeros(len(x))
        for i in range(len(x)):
            dg[i] = self.d_inparg(x, y, t).dot(n[i])
        dg *= dv
        return dg

class kachanovska_6_3(waveform):
    #another wave from kachanovska 
    def __init__(self, inc_angle = -np.pi/2, time_shift=2):
        super().__init__(inc_angle, time_shift)
        
    def profile(self, z):
        return -np.sin(8*z)/(1+np.exp(-20*z))
    
    def dprofile(self, z):
        tmp1 = (1 + np.exp(-20*z))
        return -((tmp1*8*np.cos(8*z))+np.sin(8*z)*np.exp(-20*z))/(tmp1)**2
    
    def td_deriv(self, x, y, t, n):
        dv = self.dprofile(self.inparg(x, y, t))
        dg = np.zeros(len(x))
        for i in range(len(x)):
            dg[i] = self.d_inparg(x, y, t).dot(n[i])
        dg *= dv
        return dg

class sine_pulse(waveform):
    def __init__(
                 self, 
                 inc_angle = 0, 
                 time_shift=0, 
                 r0 = np.array([0,0]),
                 src_type = 'planewave',
                 alpha = 0.5, 
                 width = 20, 
                 cycles = 1,
                ):
        super().__init__(time_shift=time_shift, 
                         inc_angle=inc_angle,
                         r0 = r0,
                         src_type=src_type)
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

class gauss_mod_sine_pulse(waveform):
    def __init__(
                self,
                inc_angle=0,
                time_shift=0, 
                r0 = np.array([0,0]),
                src_type = 'planewave',
                d = 0.075,
                w0 = 50.6
                ):
        super().__init__(inc_angle, 
                         time_shift,
                         r0 = r0,
                         src_type=src_type)
        self.d = d
        self.w0 = w0
      
    def stepfn(self, z):
        mask = (z > 0)
        arr = np.ones(len(z))
        return mask*arr
    
    def profile(self, z):
        stp = self.stepfn(z) - self.stepfn(z - 2*self.time_shift)
        squig = np.cos(self.w0*(z - self.time_shift))
        ex = np.exp(-((z-self.time_shift)**2)/(4*self.d**2))
        return (stp*squig*ex).real
    
    def td_deriv(self, x, y, t, n):
        z = self.inparg(x, y, t)
        dv = self.profile(z)*(-1j*self.w0 - 2*((z - self.time_shift)/(4*self.d**2)))
        dg = 1j*np.zeros(len(x))
        for i in range(len(x)):
            dg[i] = self.d_inparg(x, y, t).dot(n[i])
        dg *= dv
        return dg
        
class sinewave(waveform):
    def __init__(
                 self, 
                 inc_angle = 0, 
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
                 inc_angle = 0, 
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
