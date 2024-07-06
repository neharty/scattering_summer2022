import numpy as np
from scipy.special import jv, jve, jvp, hankel1, h1vp, hankel1e
from scipy.signal import convolve
from scipy.linalg import toeplitz, matmul_toeplitz, circulant
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
from numpy.fft import fft, ifft
from scipy.fft import fftfreq, rfft, irfft, rfftfreq, fftn, fftshift, ifftshift
from multiprocess import Process, Pool, cpu_count
import copy
import gc
import uuid
from Ncyl.scython.wrapper import PyScat
from dataclasses import dataclass, field
from Ncyl.v2.fds import cyl, freq_domain_scattering
from dataclasses import dataclass, field
#from tests.cpp_tests.reduced.wrapper import PyScat as PyScatC

@dataclass
class time_domain_scattering:
    cyls: object
    waveform: object
    end_time: float
    use_tmatrix: bool=True
    fft_tol: float = 1e-10
    sampnum: int = -1
    modes: int = -1
    mode_start: int = 1
    mode_tol: float = 1e-10
    use_par: bool = True
    cpus: int = -1
    use_conj: bool = True
    use_foldy_lax: bool= False
    timeint: object = None
    use_disk: bool = False
    save_coeffs: bool = False
    wvargs: dict = field(default_factory=dict)
    scargs: dict = field(default_factory=dict)
  
    def __post_init__(self):
        self.fl = self.use_foldy_lax
        self.wvp = self.waveform.profile
        self.fftidx = []
        if self.cpus < 0:
           self.cpus = cpu_count()
        self.coeff_mat = None
        self.use_coeffs = False #changed automatically after 1 computation is done
        
        # create 0-indexed labels for cylinders
        i = 0
        for c in self.cyls:
            c.label = i
            i+=1
            
        self.labels = [c.label for c in self.cyls]
        
        self.u_sc = None # initialize to prevent repeat computations
        
        self.init_vars()
        
    def init_vars(self):
        if self.use_disk:
            import os
            outpath = os.getcwd()
            self.outpath = outpath+'/tmp/'
            print('writing rhs data to ', self.outpath)
            self.uid = str(uuid.uuid4()) #uid for disk writing
            os.makedirs(self.outpath, exist_ok=True)
            
        #setup cqm variables
        self.dt = self.end_time/self.sampnum
        self.times = np.linspace(0, self.end_time, self.sampnum+1)
        eps = np.finfo(float).eps # machine epsilon
        
        if self.timeint is None:
            #BDF2
            self.lam = max(self.dt**(3/self.sampnum), eps**(1/(2*self.sampnum)))
        else:
            #IRK methods
            self.lam = max(self.dt**((self.timeint.order + 1)/self.sampnum), 
                           eps**(1/(2*self.sampnum)))
            
        zN = np.exp(-2*1j*np.pi*np.arange(0, self.sampnum+1)/(self.sampnum+1))
        self.zs = zN*self.lam
        
        if self.timeint is None:
            gm = lambda z : 0.5*((z**2)-(4*z)+3) 
            self.sl = gm(self.zs)/self.dt
            self.ks = 1j*self.sl
        else:
            blist = self.timeint.b.tolist()
            bmat = np.array([blist for b in blist])
            #gm = lambda z : self.timeint.A + (z/(1-z))*bmat
            invmat = np.linalg.inv(self.timeint.A)
            gm = lambda z : invmat@(np.eye(len(invmat)) - z*(bmat@invmat))
            #gminv = [np.linalg.inv(gm(z)) for z in self.zs]
            self.gminv = np.array([gm(z) for z in self.zs])
            peig = [np.linalg.eig(Ainv) for Ainv in self.gminv]
            self.evals = np.array([tup[0] for tup in peig])
            self.pmats = np.array([tup[1] for tup in peig])
            self.pinvmats = np.array([np.linalg.inv(p) for p in self.pmats])     
        
        self.lammat = self.lam**np.arange(0, len(self.times))
    
    def set_timeshift(self):
        if self.waveform.time_shift == 0:
            if self.waveform.src_type == 'planewave':
                #dot product
                incvect = [np.cos(self.waveform.inc_angle), np.sin(self.waveform.inc_angle)]
                pos = [c.pos for c in self.cyls]
                projmag = [np.dot(p, incvect) for p in pos]
                maxcyl = self.cyls[np.argmax(projmag)]
                projmagmax = np.max(projmag)
                self.waveform.time_shift = self.waveform.window_length + projmagmax + maxcyl.radius + self.waveform.lag
            else:
                self.waveform.time_shift = self.waveform.windwow_length + self.waveform.lag
    
    def mode_select(self, x, y):
        #make distance matrix
        dist_matrix = np.zeros((len(self.cyls), len(self.cyls))) 
        for i in self.labels:
            for j in self.labels:
                if i == j: 
                    dist_matrix[i, j] = np.inf
                else:
                    dist_matrix[i, j] = np.linalg.norm(self.cyls[i].pos - self.cyls[j].pos)

        #find closest 2 circles
        minidx = np.unravel_index(
            dist_matrix.argmin(), 
            dist_matrix.shape
            )
        print(minidx, dist_matrix[minidx])
        auxcyls = [
            copy.deepcopy(self.cyls[minidx[0]]), 
            copy.deepcopy(self.cyls[minidx[1]])
            ]
        
        #solve the problem once
        Nmodeprev = self.mode_start
        tdsc = time_domain_scattering(
                                     auxcyls,
                                     self.waveform,
                                     self.end_time,
                                     modes=Nmodeprev,
                                     sampnum=self.sampnum,
                                     timeint=self.timeint,
                                     use_par=self.use_par,
                                     scargs=self.scargs,
                                     use_cython = self.use_cython,
                                     use_cpp=self.use_cpp
                                     )
        Uscprev = tdsc.make_U_sc(x, y)
        
        #solve the problem with increasing mode numbers
        err = np.inf
        while err > self.mode_tol:
            print(Nmodeprev, err)
            auxcyls[0].sum_index = None
            auxcyls[1].sum_index = None
            Nmodecurr = Nmodeprev + 1
            tdsc = time_domain_scattering(
                                         auxcyls,
                                         self.waveform,
                                         self.end_time,
                                         modes=Nmodecurr,
                                         sampnum=self.sampnum,
                                         timeint=self.timeint,
                                         use_par=self.use_par,
                                         scargs=self.scargs,
                                         use_cpp=self.use_cpp,
                                         use_cython=self.use_cython
                                         )
            Usccurr = tdsc.make_U_sc(x, y)
            err = np.amax(np.abs(Usccurr - Uscprev))
            Uscprev = Usccurr
            Nmodeprev = Nmodecurr
        
        return Nmodecurr
    
    def td_bdy(self, cyl, c=0):
        bc = cyl.bc
        n = 2*self.modes + 1

        N = len(self.times)
        r = cyl.radius
        p = np.array(cyl.pos)
        bc = cyl.bc
        ang = np.linspace(0, 2*np.pi, num=n, endpoint=False)
        pts = np.array([[p[0] + r*np.cos(a), p[1] + r*np.sin(a)] for a in ang])
        
        #initialize arrays
        #use only for plane waves
        if self.use_tmatrix:
            g = np.zeros((n, N))
            for i in range(N):
                t = self.times[i]
                g[:, i] = self.waveform.td_wave(pts[:, 0], pts[:, 1], t+c*self.dt)
            return g
        else:
            if bc == 'i':
                ng = np.zeros((n, N))
                dg = np.zeros((n, N))
            else:
                g = np.zeros((n, N))
            
            for i in range(N):
                t = self.times[i]
                if bc == 'd':
                    g[:, i] = self.waveform.td_wave(pts[:, 0], pts[:, 1], t+c*self.dt)
                elif bc == 'n':
                    dirvect = (pts[:] - p)/r
                    g[:, i] = self.waveform.td_deriv(pts[:, 0], pts[:, 1], t+c*self.dt, dirvect)
                elif bc == 'i':
                    #neumann part
                    dirvect = (pts[:] - p)/r
                    ng[:, i] = self.waveform.td_deriv(pts[:, 0], pts[:, 1], t+c*self.dt, dirvect)
                    #dirichlet part
                    dg[:, i] = self.waveform.td_wave(pts[:, 0], pts[:, 1], t+c*self.dt)
            if bc == 'i':
                return ng, dg
            else:
                return g
        
    def ptsrc_init(self, c):
        #initialize the time profile for the point source
        tshift = self.waveform.time_shift
        ztf = self.ztrans(self.waveform.profile(self.times - tshift + c*self.dt))
        return ztf
    
    def fd_inc_wave(self, x, y, coeff):
        r0 = self.waveform.r0
        dist = lambda x, y: np.sqrt((x - r0[0])**2 + (y-r0[1])**2)
        grnmode = lambda x, y: coeff*1j*hankel1(0, self.ks*dist(x, y))/4
        return grnmode(x, y)
    
    def fd_bdy(self, cyl, coeffs):
        bc = cyl.bc
        if self.balanced_system:
            n = 2*self.modes + 1
        else:
            n = 2*self.modes
        N = len(self.times)
        r = cyl.radius
        p = np.array(cyl.pos)
        bc = cyl.bc
        ang = np.linspace(0, 2*np.pi, num=n, endpoint=False)
        pts = np.array([[p[0] + r*np.cos(a), p[1] + r*np.sin(a)] for a in ang])
        
        #initialize arrays
        #use only for pointsource waves
        if bc == 'i':
            ng = 1j*np.zeros((n, N))
            dg = 1j*np.zeros((n, N))
        else:
            g = 1j*np.zeros((n, N))
        
        #z transform the time signal
        r0 = self.waveform.r0
        dist = lambda x, y: np.sqrt((x - r0[0])**2 + (y-r0[1])**2)
        grnmodes = lambda x, y, i: coeffs[i]*1j*hankel1(0, self.ks[i]*dist(x, y))/4
        dgrnmodes = lambda x, y, i: coeffs[i]*1j*h1vp(0, self.ks[i]*dist(x, y))*self.ks[i]/(4*dist(x,y))
        
        for i in range(N):
            if bc == 'd':
                g[:, i] = grnmodes(pts[:, 0], pts[:, 1], i)
            elif bc == 'n':
                dirvect = (pts[:] - p)/r
                dotpdct = np.zeros(len(pts))
                for j in range(len(pts)):
                    ptmp = np.array([pts[j, 0]-r0[0], pts[j, 1]-r0[1]])
                    dotpdct[j] = ptmp.dot(dirvect[j])                    
                g[:, i] = dgrnmodes(pts[:, 0], pts[:, 1], i)*dotpdct
            elif bc == 'i':
                #neumann part
                dirvect = (pts[:] - p)/r
                dotpdct = np.zeros(len(pts))
                for j in range(len(pts)):
                    ptmp = np.array([pts[j, 0] - r0[0], pts[j, 1]-r0[1]])
                    dotpdct[j] = ptmp.dot(dirvect[j])  
                ng[:, i] = dgrnmodes(pts[:, 0], pts[:, 1], i)*dotpdct
                
                #dirichlet part
                dg[:, i] = grnmodes(pts[:, 0], pts[:, 1], i)
        if bc == 'i':
            return dg, ng
        else:
            return g
    
    def ztrans(self, x):
        #z-transform of a time sequence
        return fft(self.lammat*x)
        
    def ztrans_bdy_pw(self, cyl, c=0):
        # z transform the boundary data for a planewave
        # z-transform the boundary condition
        if self.use_tmatrix:
            g = self.td_bdy(cyl, c) #discretize bdy
            rhs = self.ztrans(g)   
            return rhs
        else:
            if cyl.bc == 'i':
                ng, dg = self.td_bdy(cyl, c) #discretize bdy
                rhs_d = self.ztrans(dg)
                rhs_n = self.ztrans(ng)
                return (rhs_d, rhs_n)
            else:
                g = self.td_bdy(cyl, c) #discretize bdy
                rhs = self.ztrans(g)   
                return rhs
    
    def make_zbc(self, cyl, zdat):
        #form the z-domain, perhaps coupled, boundary conditions
        if self.use_tmatrix:
            return -zdat
        else:
            bc = cyl.bc
            imp = cyl.impedance
            if bc == 'i':
                z_d, z_n = zdat
                z_d *= self.sl*imp(self.ks)
                return -z_n - z_d 
            else:
                return -zdat
    
    def fourier_coeffs_bdy(self, v):
        #project the vector onto the fourier basis using ffts
        n = 2*self.modes + 1

        if self.use_conj:
            N = len(self.times)//2 +1
        else:
            N = len(self.times)
        
        #for using foldy lax approx
        if not self.fl:
            g = 1j*np.zeros((n, N))
        else:
            # for foldy-lax approximations
            # only uses 1 mode -- incorrect for Neumann case
            g = 1j*np.zeros(N)
                
        for i in range(N):
            if self.fl:
                gtmp = fftshift(fft(v[:, i]))/n
                g[i] = gtmp[n//2]
                #sparsify the rhs
                tmp = np.abs(g[i])
                if np.max(tmp) < self.fft_tol:
                    g[i] = 0
                    return g
            else:
                g[:, i] = fftshift(fft(v[:, i]))/n
                #sparsify the rhs
                tmp = np.abs(g[:, i])
                if np.max(tmp) < self.fft_tol:
                    g[:, i] = np.zeros(n)
                    if self.use_conj:
                        g = g[:, :i+1]
                        return g 
        return g
    
    def diag_loop(self, v):
        #diagonalization loop
        gret = [None for c in self.timeint.c]
        for i in range(len(self.timeint.c)):
            gtmp = 1j*np.zeros(v[i].shape)
            for j in range(len(self.timeint.c)):
                gtmp += self.pinvmats[:, i, j]*v[j]
            gret[i] = gtmp
        return gret
    
    def diag_bc(self, cyl, v_coup):
        #uncouple the BCs, given the coupled data
        bc = cyl.bc
        
        gret = [None for c in self.timeint.c]
        if self.use_tmatrix:
            for i in range(len(self.timeint.c)):
                gret[i] = self.make_zbc(cyl, self.diag_loop(v_coup)[i])
        else:
            if bc == 'i':
                v_d = [None for c in self.timeint.c]
                v_n = [None for c in self.timeint.c]
                for i in range(len(self.timeint.c)):
                    v_d[i] = v_coup[i][0]
                    v_n[i] = v_coup[i][1]
                gdtmp = self.diag_loop(v_d)
                gntmp = self.diag_loop(v_n)
                for i in range(len(self.timeint.c)):
                    self.sl = self.evals[:, i]/self.dt
                    self.ks = 1j*self.sl
                    gret[i] = self.make_zbc(cyl, (gdtmp[i], gntmp[i]))
            else:
                for i in range(len(self.timeint.c)):
                    gret[i] = self.make_zbc(cyl, self.diag_loop(v_coup)[i])
        
        return gret
    
    def fourier_coeffs_bdy_ms(self, v):
        #return the fourier coefficients for a multistep method
        g = [1j*np.zeros(v[0].shape) for c in self.timeint.c]
        for j in range(len(self.timeint.c)):
            g[j] = self.fourier_coeffs_bdy(v[j])
        return g
    
    def fzt_boundary_1cyl(self, cyl):
        #generates the fourier coefficients for 1 circle
        #depending on the timestepping method used
        
        if self.waveform.src_type == 'planewave':
            if self.timeint == None:
                #BDF2
                #z transform td data
                zdat = self.ztrans_bdy_pw(cyl)
                
                #form the z-domain bcs
                zbc = self.make_zbc(cyl, zdat)
                
                #make fourier coefficients
                fcs = self.fourier_coeffs_bdy(zbc)
                
                #set max cyl index
                cyl.maxidx = fcs.shape[1]
                #print(fcs.shape[1])
                return fcs 
            else:
                #IRK methods
                zbc = [None for c in self.timeint.c]
                for i in range(len(self.timeint.c)):
                    #z-transform the time-domain boundary data
                    c = self.timeint.c[i]
                    zbc[i]= self.ztrans_bdy_pw(cyl, c)
                
                #make diagonalized bcs
                dbc = self.diag_bc(cyl, zbc)
                
                #return  fourier coeffs
                fcs = self.fourier_coeffs_bdy_ms(dbc)
                
                #compute the max idx for the cyl
                maxidxs = [fcs[i].shape[1] for i in range(len(self.timeint.c))]
                cyl.maxidx = max(maxidxs)
                #print(max(maxidxs))
                return fcs
        else:
            #wave is pointsource
            if self.timeint == None:
                #BDF2 algorithm
                #z transform td data (time profile)
                zdat = self.ptsrc_init(0)
                
                #form freq-domain BCs with hankel fns
                fds = self.fd_bdy(cyl, zdat)
                
                #make freq-domain bcs
                zbc = self.make_zbc(cyl, fds)
                
                #make fourier coefficients
                fcs = self.fourier_coeffs_bdy(zbc)
                
                #set max cyl index
                cyl.maxidx = fcs.shape[1]
                #print(fcs.shape[1])
                return fcs
                
            else:
                #IRK methods
                #z transform incident td data
                zdat = [None for c in self.timeint.c]
                for i in range(len(self.timeint.c)):
                    #z-transform the time-domain boundary data
                    c = self.timeint.c[i]
                    zdat[i] = self.ptsrc_init(c)
                
                #diagonalize z-domain incident data
                diag_zdat = self.diag_loop(zdat)
                
                #form freq-domain bcs with hankel fns
                zdat = [None for c in self.timeint.c]
                for i in range(len(self.timeint.c)):
                    self.sl = self.evals[:, i]/self.dt
                    self.ks = 1j*self.sl
                    zdat[i] = self.make_zbc(cyl, self.fd_bdy(cyl, diag_zdat[i]))
                
                #return  fourier coeffs
                fcs = self.fourier_coeffs_bdy_ms(zdat)
                
                #compute the max idx for the cyl
                maxidxs = [fcs[i].shape[1] for i in range(len(self.timeint.c))]
                cyl.maxidx = max(maxidxs)
                #print(max(maxidxs))
                return fcs
    
    def zeropad_rhs(self, mat):
        m, n = mat.shape
        if n < self.rhs_maxidx:
            newmat = 1j*np.zeros((m, self.rhs_maxidx))
            newmat[:, :n] = mat[:, :]
            return newmat
        else:
            return mat
    
    def fzt_boundary_allcyls(self):
        #only works for BDF2
        def writeout(c):
            tmp = self.fzt_boundary_1cyl(c)
            outstr = self.outpath+str(c.label)+'_'+self.uid+'.rhs'
            np.savetxt(outstr, tmp)
            
        if self.use_disk:
            if self.use_par:
                pool = Pool(processes=self.cpus)
                pool.map(writeout, [c for c in self.cyls])
                pool.close()
                pool.join()
            else:
                for c in self.cyls:
                    writeout(c)
        else:
            if self.use_par:
                pool = Pool(processes=self.cpus)
                rhstmp = pool.map(self.fzt_boundary_1cyl, [c for c in self.cyls])
                pool.close()
                pool.join()
                for c in self.cyls:
                    c.maxidx = rhstmp[c.label].shape[1]
            else:
                rhstmp = [self.fzt_boundary_1cyl(c) for c in self.cyls]
            self.rhs_maxidx = max([c.maxidx for c in self.cyls])
            rhstmp = [self.zeropad_rhs(m) for m in rhstmp]
            self.rhsmat = [None for n in range(self.rhs_maxidx)]
            for i in range(self.rhs_maxidx):
                if self.fl:
                    self.rhsmat[i] = np.hstack([rhstmp[c][i] for c in range(len(self.cyls))])
                else:
                    self.rhsmat[i] = np.hstack([rhstmp[c][:, i] for c in range(len(self.cyls))])
            self.rhsmat = np.array(self.rhsmat)
            if self.save_coeffs and self.coeff_mat is None:
                self.coeff_mat = 1j*np.zeros(self.rhsmat.shape)
   
    def make_time_harmonic_soln(self, x, y):
        if not self.use_coeffs:
            self.fzt_boundary_allcyls()
        self.u_sc = 1j*np.zeros((*x.shape, len(self.times)))
        klen = range(len(self.ks))
        cpus = self.cpus
        
        if self.use_conj:
            self.ks = self.ks[:self.rhs_maxidx]
        
        def comp(idx):
            if self.use_disk:
                rhs = []
                for c in self.cyls:
                    instr = self.outpath+str(c.label)+'_'+self.uid+'.rhs'
                    tmp = np.loadtxt(instr, dtype=np.complex128)
                    rhs += [tmp[:, idx]]
                rhs = np.hstack(rhs)
            else:
                rhs = self.rhsmat[idx]
            
            if np.linalg.norm(rhs) < self.fft_tol:
                return 1j*np.zeros(x.shape)
            
            sc = freq_domain_scattering(
                copy.deepcopy(self.cyls), 
                rhs, 
                self.ks[idx],
                use_tmatrix=self.use_tmatrix,
                sum_index=self.modes,
                save_coeffs = self.save_coeffs,
                **self.scargs
            )
            print('wavenumber index ' + str(idx)+ ' / '+str(self.rhs_maxidx))
            if self.save_coeffs and not self.use_coeffs:
                utmp, self.coeff_mat[idx] = sc.make_u_sc(x, y)
            elif self.save_coeffs and self.use_coeffs:
                utmp, self.coeff_mat[idx] = sc.make_u_sc(x, y, self.coeff_mat[idx])
            else:
                utmp = sc.make_u_sc(x, y)
            return utmp
        
        if self.use_par:
            #compute using the pool
            pool = Pool(processes=cpus)
            tmp = pool.map(comp, range(len(self.ks)))
            pool.close()
            pool.join()
            for i in range(len(tmp)):
                self.u_sc[..., i] = tmp[i]
            del tmp
        else:
            #serial algorithm
            for i in range(len(self.ks)):
                self.u_sc[..., i] = comp(i)
        
        #copy solutions as conjugates for negative frequency components
        if self.use_conj:
            for n in range(self.sampnum//2):
                self.u_sc[..., -1-n] = self.u_sc[..., n+1].conjugate()
        
        return self.u_sc
    
    def fzt_boundary_allcyls_highorder(self):
        def writeout(c):
            tmp = self.fzt_boundary_1cyl(c)
            for j in range(len(self.timeint.c)):
                outstr = self.outpath+str(c.label)+'_'+str(j)+'_'+self.uid+'.rhs'
                np.savetxt(outstr, tmp[j])
            
        if self.use_disk:
            if self.use_par:
                pool = Pool(processes=self.cpus)
                pool.map(writeout, [c for c in self.cyls])
                pool.close()
                pool.join()
            else:
                for c in self.cyls:
                    writeout(c)
        else:
            if self.use_par:
                pool = Pool(processes=self.cpus)
                rhstmp = pool.map(self.fzt_boundary_1cyl, [c for c in self.cyls])
                pool.close()
                pool.join()
            else:
                rhstmp = [self.fzt_boundary_1cyl(c) for c in self.cyls]
            for c in self.cyls:
                c.maxidx = [rhstmp[c.label][i].shape[1] for i in range(len(self.timeint.c))]
            self.rhs_maxidx_matrix = [c.maxidx for c in self.cyls]
            self.rhs_maxidx = np.amax(self.rhs_maxidx_matrix)
            for j in range(len(self.timeint.c)):
                for c in range(len(self.cyls)):
                    rhstmp[c][j] = self.zeropad_rhs(rhstmp[c][j])
            self.rhsmat = [[None for n in range(self.rhs_maxidx)] for c in range(len(self.timeint.c))]
            for j in range(len(self.timeint.c)):
                for i in range(self.rhs_maxidx):
                    self.rhsmat[j][i] = np.hstack([rhstmp[c][j][:, i] for c in range(len(self.cyls))])
            if self.save_coeffs and self.coeff_mat is None:
                self.coeff_mat = [1j*np.zeros(self.rhsmat[j].shape) for j in range(len(self.timeint.c))]
    
    def highorder_solve(self, idx, x, y):
        ks = 1j*self.evals[idx]/self.dt
        wmat = [None for k in ks]
        
        for i in range(len(ks)):
            wvidx = len(self.timeint.c)*idx + i
            print('wavenumber index ' + str(wvidx) + ' / ' + str((self.rhs_maxidx)*len(self.timeint.c) - 1))
            if self.use_disk:
                rhs = []
                for c in self.cyls:
                    instr = self.outpath+str(c.label)+'_'+str(i)+'_'+self.uid+'.rhs'
                    tmp = np.loadtxt(instr, dtype=np.complex128)
                    rhs += [tmp[:, idx]]
                rhs = np.hstack(rhs)
            else:
                rhs = self.rhsmat[i][idx]

            if np.linalg.norm(rhs) < self.fft_tol:
                wmat[i] = 1j*np.zeros(x.shape)
            else:
                sc = freq_domain_scattering(
                        copy.deepcopy(self.cyls), 
                        rhs, 
                        ks[i],
                        use_tmatrix=self.use_tmatrix,
                        sum_index=self.modes,
                        save_coeffs = self.save_coeffs,
                        **self.scargs
                    )
                if self.save_coeffs and not self.use_coeffs:
                    wmat[i], self.coeff_mat[i][idx] = sc.make_u_sc(x, y)
                elif self.save_coeffs and self.use_coeffs:
                    wmat[i], self.coeff_mat[i][idx] = sc.make_u_sc(x, y, self.coeff_mat[i][idx])
                else:
                    wmat[i] = sc.make_u_sc(x, y)

        rmat = 1j*np.zeros(wmat[0].shape)
        for i in range(len(wmat)):
            rmat += self.pmats[idx][-1, i]*wmat[i]

        return self.zs[idx]*rmat
    
    def make_time_harmonic_soln_highorder(self, x, y):
        if not self.use_coeffs:
            self.fzt_boundary_allcyls_highorder()
        self.u_sc = 1j*np.zeros((*x.shape, len(self.times)))
        cpus = self.cpus
        if self.use_conj:
            idxs = np.arange(self.rhs_maxidx)
        else:
            idxs = np.arange(len(self.times))
        
        def comp(idx):
            return self.highorder_solve(idx, x, y)
        
        if self.use_par:
            #spawn process pool
            pool = Pool(processes = cpus)
            #compute using the pool
            tmp = pool.map(comp, idxs)
            pool.close()
            pool.join()
            for i in idxs:
                self.u_sc[..., i] = tmp[i]
            del tmp
        else:
            #serial algorithm
            for i in idxs:
                self.u_sc[..., i] = self.highorder_solve(i, x, y)
        
        #copy solutions as conjugates for negative frequency components
        if self.use_conj:
            for n in range(self.sampnum//2):
                self.u_sc[..., -1-n] = self.u_sc[..., n+1].conjugate()
        
        return self.u_sc
    
    def make_U_sc(self, x, y):
        #set waveform variables
        self.set_timeshift()
        
        #select Nmodes
        if self.modes is None:
            self.modes=self.mode_select(x, y)
        
        if self.timeint is None:
            # BDF2
            if self.u_sc is None:
                self.make_time_harmonic_soln(x, y)
        else:
            # IRK timestepping
            if self.u_sc is None:
                self.make_time_harmonic_soln_highorder(x, y)
            
        if self.save_coeffs:
            self.use_coeffs = True
        
        self.U_sc = (1/self.lammat)*ifft(self.u_sc, axis=-1)
        self.U_sc = self.U_sc.real 
        return self.U_sc

    def make_U_inc(self, x, y):
        if self.waveform.src_type == 'planewave':
            self.waveform.td_wave(x, y, self.times)
        
        #for the point source solution at a point
        #form z-domain solution
        if self.timeint is None:
            #hacky solution
            useconjtmp = self.use_conj
            self.use_conj = True
            self.init_vars()
            self.use_conj = useconjtmp
            self.init_vars()
            
            coeffs = self.ptsrc_init(0)
            w = self.fd_inc_wave(x, y, coeffs)
            self.U_inc = (1/self.lammat)*ifft(w)
        else:
            wmat = [None for c in self.timeint.c]
            
            zdat = [None for c in self.timeint.c]
            for i in range(len(self.timeint.c)):
                #z-transform the time-domain boundary data
                c = self.timeint.c[i]
                zdat[i] = self.ptsrc_init(c)
            
            #diagonalize z-domain incident data
            diag_zdat = self.diag_loop(zdat)
            
            for i in range(len(self.timeint.c)):
                self.ks = 1j*self.evals[:, i]/self.dt
                c = self.timeint.c[i]
                wmat[i] = self.fd_inc_wave(x, y, diag_zdat[i])

            rmat = 1j*np.zeros(wmat[0].shape)
            for i in range(len(self.timeint.c)):
                rmat += self.pmats[:, -1, i]*wmat[i]

            rmat = self.zs*rmat
            self.U_inc = (1/self.lammat)*ifft(rmat)
        
        return self.U_inc.real

    def make_U(self, x, y):
        self.make_U_sc(x, y)
        
        self.U = np.zeros(self.U_sc.shape)
        for i in range(len(self.times)):
            if self.waveform.src_type == 'planewave':
                self.U[..., i] = self.U_sc[:, :, i] + \
                    (freq_domain_scattering.mask(self, x, y)*\
                     self.waveform.td_wave(x, y, self.times[i]))
            else:
                #point source wave
                self.U[..., i] = self.U_sc[..., i] + \
                        self.make_U_inc(x, y)
                
        return self.U

    
