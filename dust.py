import numpy as np
import constants as c
import fct as f
import azimuthal_profile as az
import functools as ft
class Dust:

    def __init__(self):
        self.v_f = 500.                #fragmentation velocity [cm/s]
        self.a_min = 1.0e-5          #monomer radius (smallest grain size) [cm]
        self.a_max = None           #maximum aggregate size assuming fragmentation limit
        self.rho_s = 1.4               #dust particle solid density [g/cm3]
        self.sig = None # total dust surface density [g/cm2]
        self.dsig = None # dust surface density distribution n(a)*m*a [g/cm3]
        self.Nf = 300#1500 #number of bins

        self.model = None #model of the dust background distribution
        space_a_min = 0.025e-4
        space_a_max = 1000.0


        self.a_grid = np.logspace(np.log10(self.a_min), np.log10(space_a_max), num=self.Nf, dtype=float) # logarithmic grid containing all the dust particle size bins
        self.dloga = np.diff(np.log(self.a_grid))[0] #logarithmig binspacing

        self.a = None           #particle size of a single dust particle from tge background distribution
        self.h = None           #dust scale height for particles of size self.a

        self.mj = np.zeros_like(self.a_grid) #mass of the dust particles
        self.nj = np.zeros_like(self.a_grid) #number density of grains of size j at altizute z
        self.ts = np.zeros_like(self.a_grid) #stopping time for each particle size at height z
        self.ts_mp = np.zeros_like(self.a_grid) #stopping time at the midplane for each particle size
        self.h = np.zeros_like(self.a_grid) #dust scale height for each particle size (based on midplane stopping time)
        self.Cj = np.zeros_like(self.a_grid)          #collision rates
        self.Cj_hat = np.zeros_like(self.a_grid) #adjusted collision rates
        self.Ctot = 0.0   #total collision rates
        self.Ctot_hat  = 0.0               #total collision rates
        self.sigj = np.zeros_like(self.a_grid) #surface density of particle size j in [g/cm3]

        self.sig_colj = np.zeros_like(self.a_grid)     #collision cross section
        self.v_rel = np.zeros_like(self.a_grid)       #relative velocity
        self.dv_bm = np.zeros_like(self.a_grid)
        self.dv_tur = np.zeros_like(self.a_grid)
        self.Pcol = 0.0 #collsion probability
        self.feps = 0.1 #group all the collsions below this mass fraction
        self.checkfeps = np.zeros_like(self.a_grid)

        self.a_grid_group = np.zeros_like(self.a_grid) #grouped particle size bins
        self.mj_group = np.zeros_like(self.a_grid) #mass of the grouped dust particles

    def initialize(self,simu):
        inargs = simu.parameters.inp_args

        self.v_f = inargs.v_f
        self.v_b = inargs.v_b
        self.a_min = inargs.a_min
        self.a_max = inargs.a_max
        self.rho_s = inargs.rho_s
        self.Nf = inargs.Nf
        self.model = inargs.dbgmodel

        if (self.model == 'SW'):
            print('sweep up model')

            self.Nf = 1

            self.a_grid = np.logspace(np.log10(self.a_min), np.log10(self.a_min), num=self.Nf, dtype=float) # logarithmic grid containing all the dust particle size bins
            self.dloga = 0.0#np.diff(np.log(self.a_grid))[0] #logarithmig binspacing

            self.mj = np.zeros_like(self.a_grid) #mass of the dust particles
            self.nj = np.zeros_like(self.a_grid) #number density of grains of size j at altizute z
            self.ts = np.zeros_like(self.a_grid) #stopping time for each particle size at height z
            self.ts_mp = np.zeros_like(self.a_grid) #stopping time at the midplane for each particle size
            self.h = np.zeros_like(self.a_grid) #dust scale height for each particle size (based on midplane stopping time)
            self.Cj = np.zeros_like(self.a_grid)          #collision rates
            self.Cj_hat = np.zeros_like(self.a_grid) #adjusted collision rates
            self.Ctot = 0.0   #total collision rates
            self.Ctot_hat  = 0.0               #total collision rates
            self.sigj = np.zeros_like(self.a_grid) #surface density of particle size j in [g/cm3]

            self.sig_colj = np.zeros_like(self.a_grid)     #collision cross section
            self.v_rel = np.zeros_like(self.a_grid)       #relative velocity
            self.dv_bm = np.zeros_like(self.a_grid)
            self.dv_tur = np.zeros_like(self.a_grid)
            self.checkfeps = np.zeros_like(self.a_grid)

            self.a_grid_group = np.zeros_like(self.a_grid) #grouped particle size bins
            self.mj_group = np.zeros_like(self.a_grid) #mass of the grouped dust particles

    def update(self,simu):
        '''
        update all the dust quantities
        '''
        #background dust model
        self.sig = simu.disk.dtg*simu.disk.gas.sig  # dust surface density [g/cm2]
        self.a_max = 3.*simu.disk.gas.sig/(2.*np.pi*simu.disk.gas.alpha*self.rho_s)*(self.v_f/simu.disk.gas.cs)**2. #Eq. (14)
        if (self.model == 'birnst2012'):
            self.dsig = az.size_distribution_recipe(a_grid=self.a_grid,sig_g=simu.disk.gas.sig,T=simu.disk.gas.T,sig_d=self.sig,alpha=simu.disk.gas.alpha,rho_s=self.rho_s,v_f=self.v_f,ret='sig')
            self.sigj = self.dsig*self.dloga #surface density for each mass bin
        elif(self.model == 'SW'):
            self.dsig = 0.0
            self.sigj = simu.disk.dtg*simu.disk.gas.sig


        self.mj = (4./3.)*np.pi*self.rho_s*self.a_grid**3.





    def update_rates(self,simu):
        '''
        update all the collsion rates based on the relative velocities
        '''
        alpha = simu.disk.gas.alpha
        Omega = simu.disk.Omega
        self.ts = np.sqrt(np.pi/8.)*(self.rho_s*self.a_grid)/(simu.disk.gas.rho*simu.disk.gas.cs) #stopping time at height z
        self.ts_mp = np.sqrt(np.pi/8.)*(self.rho_s*self.a_grid)/(simu.disk.gas.rho_mp*simu.disk.gas.cs) #stopping time at the mid-plane

        #Stokes drag regime
        mask = self.a_grid>(2.25*simu.disk.gas.lmfp)
        if self.ts[mask] != []:
            self.ts[mask] = (4.*self.a_grid[mask])/(9.*simu.disk.gas.lmfp)*self.ts[mask]
            self.ts_mp[mask] = (4.*self.a_grid[mask])/(9.*simu.disk.gas.lmfp)*self.ts_mp[mask]
        self.h = simu.disk.gas.h*np.sqrt((alpha/(alpha+Omega*self.ts_mp))*((1.+Omega*self.ts_mp)/(1.+2.*Omega*self.ts_mp))) #sclae hieghzt of dust grains in bin j
        self.nj = 1./(np.sqrt(2.*np.pi)*self.h)*(self.sigj/self.mj)*np.exp((-simu.particle.z**2.)/(2.*self.h**2)) #number ensity of grains of size bin j
        self.sig_colj = np.pi*(np.ones_like(self.sig_colj)*simu.particle.a+simu.disk.dust.a_grid)**2. #collsion cross section between tracked particle and each background particle bin


        #perform some preparation
        a1 = simu.particle.a #size of the tracked particle

        N = np.shape(self.v_rel)[0] #number of particle bins
        alpha = simu.disk.gas.alpha #alpha turbulence coefficient
        sig_h2 = 2.0e-15 #collision cross section of H2 in [cm2]
        sigma_g = simu.disk.gas.sig #gas surface density in [g/cm2]
        rs    = simu.disk.dust.rho_s   #dust solid density [g/cm3]
        T = simu.disk.gas.T #gas temperature
        mu = 2.3 #mean molecular weight
        m_p = 1.67e-24 #mass of a hydrogen atom [g]
        k_b = c.kB  # Boltzmann constant   [erg/K]
        Grav = c.G             # Gravitational constant        [cm3/(g s2)]
        m_star = c.Mstar         # Solar mass                    [g]
        AU = c.AU         # Astronomical unit             [cm]

        #derived quantities
        re    = alpha*sig_h2*sigma_g/(2.*mu*m_p) #reynolds number of the gas
        cs     = simu.disk.gas.cs           #isothermal sound speed
        x = simu.particle.r #radius

        omega = simu.disk.Omega     #keplerian frquency
        tn    = 1./omega                            #overturn time of the larges eddies (1 orbital timescale)
        ts    = tn*re**(-0.5)                       #overturn time of the smallest eddies
        vn    = np.sqrt(2./3.*alpha)*cs#np.sqrt(alpha)*cs             #large eddy velocity
        vs    = vn*re**(-0.25)                      #velocity of the smallest eddie (?)





        tau_1 = simu.particle.ts #stopping time of particel 1
        v_z_1 = simu.particle.v_settle #vertical settling velocity of particle 1
        v_r_1 = 0.0 #radial drift velocity of particle 1

        for i in range(N): #sweep through all the particle bins
            a2 = simu.disk.dust.a_grid[i]
            tau_2 = np.sqrt(np.pi/8.)*(simu.disk.dust.rho_s*a2)/(simu.disk.gas.rho*simu.disk.gas.cs)#stopping time of particle 2
            v_z_2 = -tau_2*(simu.disk.Omega)**2*simu.particle.z #vertical settling velocity of particle 2
            v_r_2 = -tau_2*simu.disk.Omega_mid*simu.disk.gas.eta*simu.particle.r*simu.disk.Omega_mid  #radial drift velocity of particle 2

            #sources of relative velocitiy
            dv_bm = f.dv_brownian(T=T,rho_s=rs,a1=a1,a2=a2) #brownian motion
            dv_tur = f.dvt_ormel(tau_1,tau_2,tn,vn,ts,vs,re) #stochastic turbulence
            dv_z = np.abs(v_z_1-v_z_2) #differential settling
            dv_r = np.abs(v_r_1-v_r_2) #differential drift

            self.v_rel[i] = np.sqrt(dv_tur**2.+dv_bm**2.+dv_z**2+dv_r**2.) #Okuzumi (2012) Eq.(16)

        ########################################################################
        self.Cj = np.ones_like(self.nj)*self.nj*self.v_rel*self.sig_colj #collsion rates for all the mass bins
        self.Cj_hat = np.ones_like(self.nj)*self.nj*self.v_rel*self.sig_colj


        self.checkfeps = (self.a_grid/simu.particle.a)**3. #need to group the collsions?
        self.Cj_hat[self.checkfeps<=self.feps] = np.ones_like(self.Cj[self.checkfeps<=self.feps])*(1./self.feps)*self.checkfeps[self.checkfeps<=self.feps]*self.Cj[self.checkfeps<=self.feps] #group collsiions when the collision partner is small

        self.a_grid_group = np.ones_like(self.a_grid)*self.a_grid
        self.a_grid_group[self.checkfeps<=self.feps] = np.ones_like(self.a_grid[self.checkfeps<=self.feps])*(self.feps)**(1./3.)*simu.particle.a #small mass bins have an increased mass because they now represent a group

        self.mj_group = (4.0/3.0)*np.pi*self.rho_s*self.a_grid_group**3.

        self.Ctot = np.sum(self.Cj) #total collsion rate
        self.Ctot_hat = np.sum(self.Cj_hat) #total collsion rate
