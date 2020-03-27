import numpy as np

class Gas:

    def __init__(self):
        self.sig = None              #gas surface density [g/cm2]
        self.alpha = 1.0e-3          #alpha turbulence parameter [1]
        self.D = None                #gas diffusivity
        self.cs = None               #gas speed of sound
        self.h = None               #gas scale height
        self.rho_gz = None           #volume density at height z
        self.rho_mp = None          #volume density at the midplane

    def update(self,simu):

        hg = self.h
        self.rho_gz = self.sig/(np.sqrt(2.*np.pi)*hg)*np.exp(-(simu.particle.z**2.)/(2.*hg**2.)) #update gas volume density at height z
