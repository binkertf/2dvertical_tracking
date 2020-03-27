
import numpy as np
import azimuthal_profile as az


import matplotlib.pyplot as plt

def vertical_iteration(z_i):
    z_ip1 = 0.0
    return z_ip1

###############################################################################
#dust size distribution input arguments, following Krijt (2016)
a_min = 1.0e-5          #monomer radius (smallest grain size) [cm]
sig_g=180.              #gas surface density [g/cm2]
dust_to_gas = 0.01      #dust to gas ratio
sig_d=dust_to_gas*sig_g # dust surface density [g/cm2]
alpha = 1.0e-3          #alpha turbulence parameter [1]
rho_s=1.4               #dust particle solid density [g/cm3]
R = 5.                  #radius [AU]
T = 280.*(R)**(-0.5)    #mid-plane temperature [K] (Kijt (2016  ))
v_f=500.                #fragmentation velocity [cm/s]
################################################################################

# grid of dust particle sizes [cm]
a_grid = np.logspace(np.log10(a_min), 0.5, num=150, dtype=float)
#surface density for dust particles of size a_grid
sig = az.size_distribution_recipe(a_grid=a_grid,sig_g=sig_g,T=T,sig_d=sig_d,alpha=alpha,rho_s=rho_s,v_f=v_f,ret='sig')



#plot the surface density vs. particle size
plt.loglog(a_grid,sig)
plt.ylim(1.0e-4,4)
plt.xlim(1.0e-5,1.0e1)
plt.show()
