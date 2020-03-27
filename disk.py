from gas import Gas
from dust import Dust
import constants as c

class Disk:
    '''
    This class creates a 'disk' object in which all the disk properties are stored.
    '''

    def __init__(self):
        self.gas = Gas()
        self.dust= Dust()
        self.dtg = 0.01      #dust to gas ratio
        self.R = 5.*c.AU     #radius [cm]
        self.T = None        #disk temperature [K]
        self.Omega = None   #keplerian frequency
