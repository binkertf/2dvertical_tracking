from gas import Gas
from dust import Dust
import numpy as np
import constants as c
import utils as ut


class Disk:
    '''
    This class creates a 'disk' object in which all the disk properties are stored.
    '''

    def __init__(self):
        self.gas = Gas()
        self.dust= Dust()
        self.r_min = 0.5*c.AU
        self.r_max = 7.0*c.AU
        self.r_N = 500

        self.z_max = 0.6*c.AU
        self.z_min = None

        self.z_N = 501
        self.z_mp = 0.0

        #1D array
        self.r_grid = None
        self.z_grid =  None

        #2D array
        self.r_spher = None #radius in spherical coordinates
        self.r_spher = None

        #2D array
        self.Omega_2d = None
        self.Omega_2d = None #local keplerian frequency

        self.Omega = 0.0
        self.Omega_mid = 0.0

        self.dtg = 0.01      #dust to gas ratio

    def initialize(self):
        self.z_min = -self.z_max
        self.z_mp = int(self.z_N/2)
        self.r_grid = np.linspace(self.r_min,self.r_max,self.r_N)
        self.z_grid =  np.linspace(self.z_min,self.z_max,self.z_N)

        self.r_spher = np.zeros((self.r_N,self.z_N)) #radius in spherical coordinates
        self.r_spher = ut.r_cyl_to_spher(self.r_spher,self.r_grid,self.z_grid)

        self.Omega_2d = np.zeros((self.r_N,self.z_N))
        self.Omega_2d = np.sqrt(c.G*c.Mstar/self.r_spher**3.) #local keplerian frequency



    def update(self,simu):
        r = simu.particle.r
        z = simu.particle.z

        self.Omega_mid = np.sqrt(c.G*c.Mstar/r**3.)
        self.Omega = self.Omega_mid*(1.+(z/r)**2.)**(-3./4.) #horizontal balance
        #self.Omega = self.Omega_mid*(1.-(3./4.)*(z/r)**2) #Takeuchi & Lin (2002) Eq. (15)
