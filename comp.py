import constants as c
import numpy as np


R = 5.0 * c.AU


a_list = ([1.0,0.1,0.0001])

a = a_list[2]

alpha = 1.0e-3

T = 280.*(R/c.AU)**(-0.5)
Omega = np.sqrt(c.G*c.Mstar/R**3.)
cs = (c.kB*T/c.mg)**0.5
rho_s = 1.4

sigma_g = 2000.0*(R/c.AU)**(-3./2.)
hg = cs/Omega
z = 0.0
rho_g = sigma_g/(np.sqrt(2.*np.pi)*hg)*np.exp(-(z**2.)/(2.*hg**2.))

t_s = np.sqrt(np.pi/8.)*(rho_s*a)/(rho_g*cs) #stopping dust_sigma_check
St = t_s*Omega

t_settle = 1./(St*Omega)



hd = hg *np.sqrt(alpha/(alpha+St)*((1.+St)/(1.+2.*St)))
print(hg)
print(hd)
