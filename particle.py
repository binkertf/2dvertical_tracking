import numpy as np
import constants as c
import fct as f



class Particle():

    def __init__(self):
        self.a = 1.0e-5     #initial particle size in cm
        self.z = 0.0       # initial altitude above the midplane
        self.x = 5.0/np.sqrt(2)*c.AU
        self.y = 5.0/np.sqrt(2)*c.AU
        self.r = np.sqrt(self.x**2+self.y**2)

        self.T = 0.0        #temperature
        self.ts  = None     #stopping time
        self.ts_mp = None  #stopping time at the midplane
        self.St = None      #Stokes number
        self.St = None      #mid-plane Stokes number
        self.v_settle  = 0.0 #vertical settling velocity
        self.v_gas_z  = 0.0 #vertical gas velocity
        self.v_gas_y  = 0.0 # gas velocity
        self.v_gas_x  = 0.0
        self.D = 0.0         # diffusivity
        self.v_eff_z = 0.0   # effective velocity
        self.v_eff_x = 0.0
        self.v_eff_y = 0.0
        self.v_z = 0.0
        self.v_y = 0.0
        self.v_x = 0.0
        self.v_r = 0.0 #radial drift velocity

        self.diffusion = True
        self.randmotion = True
        self.rad_vel = None




        self.h = None   #dust sclae height for particle with size a
        self.m = None   #mass of the particle

    def initialize(self,simu):
        inargs = simu.parameters.inp_args

        self.a = inargs.a0
        self.z = inargs.z0*c.AU
        self.r = inargs.r0*c.AU

        self.x = self.r #particle initially in the X-Z plane
        self.y = 0.0







    def update(self,simu):
        self.T = simu.disk.gas.T
        self.ts = np.sqrt(np.pi/8.)*(simu.disk.dust.rho_s*self.a)/(simu.disk.gas.rho*simu.disk.gas.cs) #stopping time
        self.ts_mp = np.sqrt(np.pi/8.)*(simu.disk.dust.rho_s*self.a)/(simu.disk.gas.rho_mp*simu.disk.gas.cs)
        self.St = self.ts*simu.disk.Omega #Stokes number
        if (self.randmotion == True):
            self.D = simu.disk.gas.D/(1.+(self.St)**2.) #dust diffusivity
        else:
            self.D = 0.0


        self.St_mid = self.ts_mp * simu.disk.Omega_mid

        if (simu.particle.rad_vel == 'Nakagawa86'):
            self.v_r = -2.*self.St*simu.disk.gas.eta*self.r*simu.disk.Omega_mid

        elif (simu.particle.rad_vel == 'T&L02'):
            self.v_r = ((1./self.St)*simu.disk.gas.v_r-simu.disk.gas.eta*self.r*simu.disk.Omega_mid)/(self.St+1./self.St) #radial drift velocity T&L 2002 Eq. (23)

        elif (simu.particle.rad_vel == 'Fabian20'):
            self.v_r = -self.St*simu.disk.gas.eta*self.r*simu.disk.Omega_mid #in the limiting case of St << 1

        #velocities
        self.v_settle  = -self.ts*(simu.disk.Omega)**2*self.z #vertical settling velocity
        self.v_gas_z = -self.D*self.z/simu.disk.gas.h**2.
        self.v_eff_z = self.v_gas_z + self.v_settle

        self.v_x = (self.v_r)*(self.x/self.r)
        self.v_y = (self.v_r)*(self.y/self.r)

        self.v_gas_x = self.D/simu.disk.gas.rho*simu.disk.gas.drho_dr*(self.x/self.r)
        self.v_gas_y = self.D/simu.disk.gas.rho*simu.disk.gas.drho_dr*(self.y/self.r)

        self.v_eff_x = self.v_x+self.v_gas_x
        self.v_eff_y = self.v_y+self.v_gas_y
        self.m = (4./3.)*np.pi*simu.disk.dust.rho_s*self.a**3. #particle mass
