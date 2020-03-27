from disk import Disk
from particle import Particle
import constants as c

import fct as f

import numpy as np
import azimuthal_profile as az
import matplotlib.pyplot as plt


class Simu:

    def __init__(self):
        self.disk = Disk()              #the background
        self.particle = Particle()      #the tracer
        self.parameters = Par()         #simulatiom paramters
        self.mode = 0                   #simulatiomn modes


    #the initialization method
    def initalize(self):
        '''
        This method initializes the simulation by generating the steday-state isothermal disk model based on the
        inital parameters and the model by Krijt (2016).
        '''

        #disk/gas model
        self.disk.T = f.temperature(self)   #mid-plane temperature [K] (Krijt (2016))
        self.disk.gas.sig = 2000.0*(self.disk.R/c.AU)**(-3./2.) #local gas durface density
        self.disk.Omega = np.sqrt(c.G*c.Mstar/self.disk.R**3.)
        self.disk.gas.cs = (c.kB*self.disk.T/c.mg)**(1./2.)
        self.disk.gas.h = self.disk.gas.cs/self.disk.Omega #gas scale height
        self.disk.gas.D = self.disk.gas.alpha*self.disk.gas.cs*self.disk.gas.h #gas diffusivity, Eq.(3)
        self.disk.gas.rho_mp = self.disk.gas.sig/(np.sqrt(2.*np.pi)*self.disk.gas.h) #midplane volume denisty

        #background dust model
        self.disk.dust.sig = self.disk.dtg*self.disk.gas.sig  # dust surface density [g/cm2]
        self.disk.dust.a_max = 3.*self.disk.gas.sig/(2.*np.pi*self.disk.gas.alpha*self.disk.dust.rho_s)*(self.disk.dust.v_f/self.disk.gas.cs)**2. #Eq. (14)
        self.disk.dust.dsig = az.size_distribution_recipe(a_grid=self.disk.dust.a_grid,sig_g=self.disk.gas.sig,T=self.disk.T,sig_d=self.disk.dust.sig,alpha=self.disk.gas.alpha,rho_s=self.disk.dust.rho_s,v_f=self.disk.dust.v_f,ret='sig')

        #self.disk.dust.dsig = f.background_disrtibution2(self)


        self.disk.dust.sigj = self.disk.dust.dsig*self.disk.dust.dloga #surface density for each mass bin
        self.disk.dust.mj = (4./3.)*np.pi*self.disk.dust.rho_s*self.disk.dust.a_grid**3.

        #init particle


    def rates(self):
        self.disk.dust.update_rates(self) #compute all the collision rates

    def update(self):
        self.disk.gas.update(self) # updates the local gas background
        self.particle.update(self)
        self.disk.dust.update(self) # updates the local dust background
        self.parameters.est_dt(self) #updates the simulation parameters (e.g. dt)


    def iter(self):
        if self.mode == 0: #full physics
            R1 = np.random.uniform(-1.,1.)
            rand = R1*np.sqrt((2./self.parameters.zeta)*self.particle.D*self.parameters.dt)
            self.particle.z = self.particle.z+self.particle.v_eff*self.parameters.dt+rand

        if self.mode == 1: #only vertical setling
            self.particle.z = self.particle.z+self.particle.v_settle*self.parameters.dt

        if self.mode == 2: #naive random walk
            R1 = np.random.uniform(-1.,1.)
            rand = R1*np.sqrt((2./self.parameters.zeta)*self.particle.D*self.parameters.dt)
            self.particle.z = self.particle.z+self.particle.v_settle*self.parameters.dt+rand

        if self.mode == 3: #particle stays at the midplane
            self.particle.z = 0.0

    def collision(self):
        '''
        determine if a collision has occured and if yes, what is the outcome
        '''
        np.random.seed() #this is neccessary here for calling the method in parallel processes

        vfrag = self.disk.dust.v_f #fragmentation velocity
        dvfrag = vfrag/5.
        vrel = self.disk.dust.v_rel #relative velocities

        #determine collision probability
        coll = False
        Pcol = 1.-np.exp(-self.disk.dust.Ctot_hat*self.parameters.dt) #collision probability within timestep dt
        R2 = np.random.uniform(0.,1.)
        if R2<=Pcol:
            #a collsion has occured
            coll = True
            # determine from which size bin the collider was
            R3 = np.random.uniform(0.,self.disk.dust.Ctot_hat)
            Csum = 0.0
            k = 0
            while (Csum<R3):
                Csum += self.disk.dust.Cj_hat[k]
                k +=1

            #ungrouped collsion partner
            ak = self.disk.dust.a_grid[k]
            mk = self.disk.dust.mj[k]

            #grouped collision partner
            ak_group = self.disk.dust.a_grid_group[k]
            mk_group = self.disk.dust.mj_group[k]

            n_group = mk_group/mk #number of grouped partikcles in group k

            #determine the outcome of the collision partner
            Pfrag = 0.0
            if (vrel[k]<(vfrag-dvfrag)):
                Pfrag=0.0 #no fragmentation
            elif (vrel[k]>=vfrag):
                Pfrag=1.0 #always fragmentation
            else:
                Pfrag=1.-(vfrag-vrel[k])/(dvfrag) #sometimes fragmentation

            #fragmentation?
            Frag = False #sticking collision
            R4  = np.random.uniform(0.,1.)
            if (R4<Pfrag):
                Frag = True #fragmentation collision

            #where does the monomer end up?

            if Frag == False: #sticking collision
                self.particle.m = self.particle.m+mk_group #sticking collsion with group
                self.particle.a = (3./(4.*np.pi)*(self.particle.m/self.disk.dust.rho_s))**(1./3.)


            #fragmentation
            if Frag == True:
                mass_ratio = mk/self.particle.m #mass ratio between colldiing particles

                if (mass_ratio >= 0.1): #fragmentation or erosion and tracker in smaller aggregate
                    af_min = self.disk.dust.a_min
                    af_max = self.particle.a #largest particle size taking part in the collsion
                    Cf = self.particle.m #constant in Eq. (13), total mass
                    self.particle.a = f.frag_distribution(self,af_min,af_max,m_tot=Cf)
                    self.particle.m = (4./3.)*np.pi*self.disk.dust.rho_s*self.particle.a**3.

                else: #erosion, tracker in larger aggregate
                    af_min = self.disk.dust.a_min
                    af_max = ak #largest particle size taking part in the fragmentation
                    Cf = self.particle.m #constant in Eq. (13), total mass
                    m_rem = self.particle.m-mk_group #mass of unfragmented agregate
                    self.particle.a = f.erosion_distribution(self,af_min,af_max,m_rem,m_tot=Cf)
                    self.particle.m = (4./3.)*np.pi*self.disk.dust.rho_s*self.particle.a**3.



                '''
                if (mass_ratio > 10.0): #erosion, tracker in smaller aggregate
                    af_min = self.disk.dust.a_min
                    af_max = self.particle.a #largest particle size taking part in the collsion
                    Cf = 2.*self.particle.m #constant in Eq. (13), total mass
                    self.particle.a = f.frag_distribution(self,af_min,af_max,m_tot=Cf)
                    self.particle.m = (4./3.)*np.pi*self.disk.dust.rho_s*self.particle.a**3.

                elif (mass_ratio < 0.1):#erosion, tracker in larger aggregate
                    af_min = self.disk.dust.a_min
                    af_max = ak #largest particle size taking part in the fragmentation
                    Cf = mk_group+self.particle.m #constant in Eq. (13), total mass
                    m_rem = self.particle.m-2.*mk_group #mass of unfragmented agregate
                    self.particle.a = f.erosion_distribution(self,af_min,af_max,m_rem,m_tot=Cf)
                    self.particle.m = (4./3.)*np.pi*self.disk.dust.rho_s*self.particle.a**3.

                else: #0.1 < mass_ratio < 10, catastrophic fragmentation
                    af_min = self.disk.dust.a_min
                    af_max = np.amin(([ak,self.particle.a])) #largest particle size taking part in the collsion

                    Cf = mk_group+self.particle.m #constant in Eq. (13), total mass
                    self.particle.a = f.frag_distribution(self,af_min,af_max,m_tot=Cf)
                    self.particle.m = (4./3.)*np.pi*self.disk.dust.rho_s*self.particle.a**3.
                '''












    def plot_dust_size_distribution(self):
        '''
        this method plots the surface density (n(a)*m*a [g/cm3]) vs. particle size distribution (a [cm])
        '''

        plt.loglog(self.disk.dust.a_grid,self.disk.dust.dsig)
        plt.ylim(1.0e-4,4)
        plt.xlim(1.0e-5,1.0e1)
        plt.show()


    def check_dust_sigma(self):
        '''
        here I check wether the integral of the dust particle size dustribution is in
        fact equal to the inital dust surafce density value.
        '''
        print('total dust surface denisty from initial parameters: ',self.disk.dust.sig ,' g/cm2')
        dust_sigma_check=np.trapz(self.disk.dust.dsig,x=np.log(self.disk.dust.a_grid))
        print('total dust surface denisty from integrating the size distribution: ',dust_sigma_check,' g/cm2')

    #@profile
    def write(self,k):


        if (k>=self.parameters.data.shape[0]):

            N = self.parameters.t_tot/self.parameters.dt
            array = np.zeros((int(2.*N),3))
            self.parameters.data=np.append(self.parameters.data,array,axis=0)

        self.parameters.data[k,0] = self.parameters.t
        self.parameters.data[k,1] = self.particle.z
        self.parameters.data[k,2] = self.particle.a



    def run(self, t_tot):
        self.parameters.t_tot = t_tot*c.yr
        self.initalize()
        self.update()

        #create data array
        N = self.parameters.t_tot/self.parameters.dt
        self.parameters.data = np.zeros((int(2.*N),3))
        self.write(k=0)

        # main loop
        j = 1
        while (self.parameters.t/c.yr)<t_tot:
            self.rates() #determin the collision rates
            self.parameters.update_dt(self) #updates the simulation parameters (e.g. dt)
            self.iter() #displace the particle vertically according to Eq. 15
            self.collision() #determine if a collision has occured
            self.update() #update all the quantities at the new particle position

            self.write(j)
            j+=1

        #delete the trailing zero entries
        self.parameters.data=self.parameters.data[:j,:]



class Par():

    def __init__(self):
        self.zeta = 1./3.
        self.f_diff = 1.0e-2
        self.f_coll = 1.0
        self.dt = None
        self.t = 0.0
        self.data = np.array([])
        self.t_tot = None
        self.mode = 0


    def est_dt(self,simu):
        '''
        make an estimeation of the time step to have an idea how large the data array needs to be
        '''
        self.dt = self.f_diff/((simu.disk.gas.alpha+simu.disk.Omega*simu.particle.ts)*simu.disk.Omega)


    def update_dt(self,simu):
        if (self.mode == 0):
            self.dt = np.amin(([self.f_diff/((simu.disk.gas.alpha+simu.disk.Omega*simu.particle.ts)*simu.disk.Omega),self.f_coll/simu.disk.dust.Ctot_hat]))
            self.t = self.t+self.dt
            #v = self.f_diff/((simu.disk.gas.alpha+simu.disk.Omega*simu.particle.ts)*simu.disk.Omega)
            #w = self.f_coll/simu.disk.dust.Ctot_hat
            #if (v>w):
                #print('changed timestep')
            #print('1: ',)
            #print('2: ',self.f_coll/simu.disk.dust.Ctot_hat)
        if (self.mode == 1):
            self.dt = 0.5/simu.disk.dust.Ctot_hat
            #self.dt = self.f_diff/((simu.disk.gas.alpha+simu.disk.Omega*simu.particle.ts)*simu.disk.Omega)
            #print(self.f_diff/((simu.disk.gas.alpha+simu.disk.Omega*simu.particle.ts)*simu.disk.Omega)/c.yr)
            #print(0.1/simu.disk.dust.Ctot_hat/c.yr)
            #print('#######')
            self.t = self.t+self.dt
