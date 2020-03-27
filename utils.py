import constants as c
import numpy as np
import scipy.integrate as integrate
from simu import Simu
import sys


def zdata(t_tot=20000., a=1.0,mode=0):
    '''
    returns
    '''

    simulation = Simu() #create a new simulation object
    simulation.particle.a=a #set the initial particle size
    simulation.mode = mode
    simulation.run(t_tot) #run the simulation for t_tot years
    hg = simulation.disk.gas.h #gas scale height

    return simulation.parameters.data[:,1]/hg, simulation.parameters.data[:,0]

def histogram(t_tot = 20000., N=20,a=1.0,norm=True,mode=0):

    bins = 100 #number of bins in the histogram
    binrange = (-3.5,3.5) #lower and upper edge of the histogram in units of the gas scale height hg
    data, weights = zdata(t_tot,a,mode) #get the simulation data for particle of size a cm over a period of T yrs
    H = np.histogram(data,bins=bins,range=binrange,weights=weights) # create histogram of the firts particle

    for i in range(N-1):    #simulate another N-1 particles
        data, weights = zdata(t_tot,a,mode) #get the simulation data of particle i
        H2 = np.histogram(data,bins=bins,range=binrange,weights=weights) #create the histogram data for particle i
        H = (H[0]+H2[0],H[1]) #add histogram data of particle i to histogram data of particles 0 to (i-1).

        #print a status bar
        sys.stdout.write('\riter ' + str(i) + ' of ' + str(N-1))
        sys.stdout.flush()

    #normalize the histogram data
    if norm == True:
        nrm = np.sum(H[0])*((binrange[1]-binrange[0])/bins)
        H = (H[0]/nrm,H[1])

    return H

def analytic(simu,z_list,h,sigma0):
    '''
    returns the analytic solution to the diffusion-advection equation based on the midplane Stokes number
    '''
    rho_list=sigma0/(np.sqrt(2.*np.pi)*h)*np.exp(-(z_list**2.)/(2.*h**2.))

    return rho_list


def semianalytic(simu,z_list,h,sigma0):
    '''
    returns the semianalytic solution to the diffusion-advection equation
    '''
    vz = np.zeros_like(z_list)
    D = np.zeros_like(z_list)
    rho_g = np.zeros_like(z_list)
    rho_d = np.zeros_like(z_list)
    ts = np.zeros_like(z_list)
    Omega = simu.disk.Omega

    rho_g = self.disk.gas.sig/(np.sqrt(2.*np.pi)*h)*np.exp(-(z_list**2.)/(2.*h**2.))
    ts = np.sqrt(np.pi/8.)*(simu.disk.dust.rho_s*simu.particle.a)/(rho_g*simu.disk.gas.cs)
    vz = -ts*Omega**2.*z_list
    D = simu.disk.gas.D/(1.+(Omega*ts)**2.)


    n = int(len(z_list)/2)
    f = vz/D
    integral = integrate.cumtrapz(f[n:],z_list[n:])
    rho_d[n+1:] = integral
    rho_d[:n] = np.flip(integral)

    rho_d = rho_g*np.exp(rho_d)
    return rho_d
