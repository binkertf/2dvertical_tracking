import numpy as np
import constants as c
import fct as f



class Particle():

    def __init__(self):
        self.a = 1.0e-5     #initial particle size in cm
        self.z = 0.0       # initial altitude above the midplane
        self.ts  = None     #stopping time
        self.ts_mp = None  #stopping time at the midplane
        self.St = None      #Stokes number
        self.v_settle  = None #vertical settling velocity
        self.v_gas  = None #vertical gas velocity
        self.D = None         # diffusivity
        self.v_eff = None   # effective velocity
        self.h = None   #dust sclae height for particle with size a
        self.m = None   #mass of the particle





    def update(self,simu):
        self.ts = np.sqrt(np.pi/8.)*(simu.disk.dust.rho_s*self.a)/(simu.disk.gas.rho_gz*simu.disk.gas.cs)
        self.ts_mp = np.sqrt(np.pi/8.)*(simu.disk.dust.rho_s*self.a)/(simu.disk.gas.rho_mp*simu.disk.gas.cs)
        self.v_settle  = -self.ts*(simu.disk.Omega)**2*self.z
        self.D = simu.disk.gas.D/(1.+(simu.disk.Omega*self.ts)**2.)
        self.v_gas = -self.D*self.z/simu.disk.gas.h**2.
        self.v_eff = self.v_gas + self.v_settle
        self.m = (4./3.)*np.pi*simu.disk.dust.rho_s*self.a**3.
