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
        self.mode = 0                   #simulation modes


    #the initialization method
    def initialize(self):
        '''
        This method initializes the simulation by generating a steday-state isothermal disk model
        '''
        self.disk.initialize() #initialize disk
        self.disk.gas.initialize() #initialize disk
        self.particle.initialize() #initialize disk
        self.disk.dust.initialize() #initialize disk

        #disk/gas model
        self.disk.gas.sig_1d = np.zeros_like(self.disk.r_grid)
        self.disk.gas.sig_1d = f.surface_density(self,self.disk.r_grid) #gas surface density

        self.disk.gas.T_2d = np.zeros((self.disk.r_N,self.disk.z_N))
        self.disk.gas.T_2d = f.temperature_2d(self,self.disk.r_grid)  #gas temperature

        self.disk.gas.cs_2d = np.zeros((self.disk.r_N,self.disk.z_N))
        self.disk.gas.cs_2d = f.soundspeed(self,self.disk.gas.T_2d)  #gas sound speed

        self.disk.gas.h_1d =  np.zeros_like(self.disk.r_grid)
        self.disk.gas.h_1d = self.disk.gas.cs_2d[:,self.disk.z_mp]/self.disk.Omega_2d[:,self.disk.z_mp] #gas scale height

        # fill the gas volume density
        self.disk.gas.rho_2d = np.zeros((self.disk.r_N,self.disk.z_N))
        for i, r in enumerate(self.disk.r_grid):
            for j, z in enumerate(self.disk.z_grid):
                hg = self.disk.gas.h_1d[i]
                self.disk.gas.rho_2d[i,j] = self.disk.gas.sig_1d[i]/(np.sqrt(2.*np.pi)*hg)*np.exp(-(z**2.)/(2.*hg**2.))





        #init particle


    def rates(self):
        self.disk.dust.update_rates(self) #compute all the collision rates

    def update(self):
        self.disk.update(self)
        self.disk.gas.update(self) # updates the local gas background
        self.particle.update(self)
        self.disk.dust.update(self) # updates the local dust background
        self.parameters.est_dt(self) #updates the simulation parameters (e.g. dt)


    def iter(self):
        if self.mode == 0: #full physics

            #z direction
            R1_z = np.random.uniform(-1.,1.)
            #print(R1_z)
            rand_z = R1_z*np.sqrt((2./self.parameters.zeta)*self.particle.D*self.parameters.dt)
            self.particle.z = self.particle.z+self.particle.v_eff_z*self.parameters.dt+rand_z

            #x direction
            R1_x = np.random.uniform(-1.,1.)
            #print(R1_x)
            rand_x = R1_x*np.sqrt((2./self.parameters.zeta)*self.particle.D*self.parameters.dt)

            self.particle.x = self.particle.x+self.particle.v_eff_x*self.parameters.dt+rand_x

            #y direction
            R1_y = np.random.uniform(-1.,1.)
            #print(R1_y)
            rand_y = R1_y*np.sqrt((2./self.parameters.zeta)*self.particle.D*self.parameters.dt)

            self.particle.y = self.particle.y+self.particle.v_eff_y*self.parameters.dt+rand_y

            #r direction
            self.particle.r = np.sqrt(self.particle.x**2+self.particle.y**2)



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



    #@profile
    def write(self,k):


        if (k>=self.parameters.data.shape[0]):

            N = self.parameters.t_tot/self.parameters.dt
            array = np.zeros((int(2.*N),5))
            self.parameters.data=np.append(self.parameters.data,array,axis=0)

        self.parameters.data[k,0] = self.parameters.t
        self.parameters.data[k,1] = self.particle.z
        self.parameters.data[k,2] = self.particle.r
        self.parameters.data[k,3] = self.particle.a
        self.parameters.data[k,4] = self.particle.T



    def run(self, t_tot):
        self.parameters.t_tot = t_tot*c.yr
        self.initialize()
        self.update()

        #create data array
        N = self.parameters.t_tot/self.parameters.dt
        self.parameters.data = np.zeros((int(2.*N),5))
        self.write(k=0)

        # main loop
        j = 1
        while (self.parameters.t/c.yr)<t_tot:
            if self.parameters.collisions == True:
                self.rates() #determin the collision rates
                self.parameters.update_dt(self) #updates the simulation parameters (e.g. dt)
                self.iter() #displace the particle vertically according to Eq. 15
                self.collision() #determine if a collision has occured
                self.update() #update all the quantities at the new particle position

            if self.parameters.collisions == False:
                self.parameters.update_dt(self) #updates the simulation parameters (e.g. dt)
                self.iter() #displace the particle vertically according to Eq. 15
                self.update() #update all the quantities at the new particle position

            self.write(j)
            j+=1

        #delete the trailing zero entries
        self.parameters.data=self.parameters.data[:j,:]

######################
    def velocity_field(self):
        r_grid = self.disk.r_grid
        z_grid = self.disk.z_grid

        r_plot,z_plot = np.meshgrid(z_grid, r_grid)

        v_r_plot = np.zeros_like(r_plot)
        v_z_plot = np.zeros_like(r_plot)


        for i,r in enumerate(r_grid):
            for j,z in enumerate(z_grid):
                self.particle.r = r
                self.particle.z = z
                #print('i,r: ',i,r)
                #print('j,z: ',j,z)
                self.update()
                v_r_plot[i,j] = self.particle.v_r
                v_z_plot[i,j] = self.particle.v_settle



        return r_plot,z_plot,v_r_plot,v_z_plot






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

        self.collisions = True


    def est_dt(self,simu):
        '''
        make an estimeation of the time step to have an idea how large the data array needs to be
        '''
        self.dt = self.f_diff/((simu.disk.gas.alpha+simu.disk.Omega*simu.particle.ts)*simu.disk.Omega)


    def update_dt(self,simu):

        if (self.collisions == True):
            t_drift = simu.particle.r/np.abs(simu.particle.v_r)

            self.dt = np.amin(([self.f_diff/((simu.disk.gas.alpha+simu.disk.Omega*simu.particle.ts)*simu.disk.Omega),self.f_coll/simu.disk.dust.Ctot_hat,self.f_diff*t_drift]))
            self.t = self.t+self.dt

        if (self.collisions == False):
            t_drift = simu.particle.r/np.abs(simu.particle.v_r)
            self.dt = np.amin(([self.f_diff/((simu.disk.gas.alpha+simu.disk.Omega*simu.particle.ts)*simu.disk.Omega),self.f_diff*t_drift]))
            self.t = self.t+self.dt
