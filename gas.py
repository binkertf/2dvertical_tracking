import numpy as np
import utils as ut
import fct as f
import constants as c

class Gas:

    def __init__(self):
        #2D array quantities
        self.rho_2d = None              #gas volume density [g/cm3]
        self.T_2d = None               #gas temperature [K]
        self.cs_2d = None               #gas speed of sound


        #1D array quantities
        self.sig_1d = None              #gas surface density [g/cm2]
        self.h_1d = None               #gas scale height

        #scalar quantities
        self.sig0 = 1700.0 #g/cm2   #surface density at 1 AU
        self.ps = -3./2.             #power law index

        self.T0 = 280.                   #temperature at 1 AU
        self.q = -1./2.             #power law index

        self.p =  self.ps-0.5*(self.q+3)                  #power law index

        self.sig_g = 0.0
        self.rho = 0.0              #gas volume density [g/cm3]
        self.T = 0.0               #gas temperature [K]
        self.cs = 0.0               #gas speed of sound
        self.h = 0.0               #gas scale height
        self.h0 = 0.0               #gas scale height at 1 AU

        self.rho_mp = 0.0          #volume density at the midplane
        self.alpha = 1.0e-3          #alpha turbulence parameter [1]
        self.D = 0.0                #gas diffusivity

        self.eta = 0.0
        self.Omega_g = 0.0          #gas rotation frequency
        self.v_r = 0.0               #radial gas velocity [cm/s]
        self.drho_dr = 0.0          #radial derivative of the gas volume density
        self.P = 0.0                #gas pressure
        self.dP_dr = 0.0            #pressure gardient in radial direction

    def initialize(self):
        pass


    def update(self,simu): #update the local quantities at (r,z)
        r = simu.particle.r
        z = simu.particle.z

        self.sig = f.surface_density(simu,r)
        self.T = f.temperature(simu,r)
        self.cs = f.soundspeed(simu,self.T)
        self.h = self.cs/simu.disk.Omega_mid
        self.h0 = self.h*(r/c.AU)**(-(self.q+3.)/2.)
        self.D = self.alpha*self.cs*self.h #gas diffusivity, Eq.(3)
        self.rho = self.sig/(np.sqrt(2.*np.pi)*self.h)*np.exp(-(z**2.)/(2.*self.h**2.)) #update gas volume density
        self.rho_mp = self.sig/(np.sqrt(2.*np.pi)*self.h) #midplane volume denisty

        self.Omega_g = simu.disk.Omega_mid*(1.+0.5*(self.h/r)**2*(self.p+self.q+0.5*self.q*(z/self.h)**2.)) #Takeuchi&Lin Eq. (7)
        #self.v_r = -2.*np.pi*self.alpha*(self.h0/c.AU)**2.*(3*self.p+2*self.q+6.+0.5*(5.*self.q+9.)*(z/self.h)**2.)*(r/c.AU)**(self.q+0.5)  #T&L2002 Eq. 11
        #self.v_r = self.v_r*c.AU/c.yr
        self.v_r = 0.0 #no radial gas velocity

        self.eta = -(self.h/r)**2*(self.p+self.q+0.5*(self.q+3.)*(z/self.h)**2.)#T&L2002 Eq. (17)

        self.P = self.rho*self.cs**2
        self.dP_dr = self.P/r*(self.p+self.q+(self.q+3)*(z**4./(4.*self.h**4.)))


        self.eta = -self.dP_dr/self.rho*(1./(r*simu.disk.Omega**2.))# Fabian 2020

        #self.eta = 1.81e-3*np.sqrt(r/c.AU) #Nakagawa 86 Eq. (1.10)
        #print(self.eta)
        self.drho_dr = self.rho*self.p/r
