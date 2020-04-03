import constants as c
import numpy as np
from numpy import sqrt
from scipy import integrate


def temperature_2d(simu,radius):
    '''
    computes the local temperature at radius r (T is vertically constant)
    '''
    radius_2d = np.transpose(np.tile(radius,(simu.disk.z_N,1)))
    return simu.disk.gas.T0*(radius_2d/c.AU)**simu.disk.gas.q

def temperature(simu,radius):
    '''
    computes the local temperature at radius r (T is vertically constant)
    '''
    return simu.disk.gas.T0*(radius/c.AU)**simu.disk.gas.q

def soundspeed(simu,temperature):
    '''
    computes the local sound speed at radius r
    '''
    return (c.kB*temperature/c.mg)**(1./2.)


def surface_density(simu,radius):
    '''
    computes the surface density at radius r
    '''

    return simu.disk.gas.sig0*(radius/c.AU)**simu.disk.gas.ps

def background_disrtibution(simu):

    a_grid1 = simu.disk.dust.a_grid
    sig = np.ones_like(a_grid1)
    sig[a_grid1>2.0e-2] = 0.0
    dloga = np.diff(np.log(a_grid1))[0]

    C = np.sum(sig*dloga)
    sig = sig*simu.disk.dust.sig/C
    return sig

def frag_distribution(simu,a_min,a_max,m_tot):
    '''
    returns the particle in which the tracked particle ends up in when the fragments
    are distributed according to eq. (13)
    '''
    if (a_min==a_max):
        return a_min

    N = 1500
    xi = 3.5

    a_coll_grid = np.logspace(np.log10(a_min), np.log10(a_max), num=N, dtype=float)
    m_grid = (4./3.)*np.pi*simu.disk.dust.rho_s*a_coll_grid**3
    nf_a = a_coll_grid**(-xi)

    dloga = np.diff(np.log(a_coll_grid))[0]
    dM = nf_a*a_coll_grid*m_grid*dloga

    Psum = np.sum(dM)
    P = 1./Psum*dM
    a_coll = np.random.choice(a_coll_grid,p=P)
    return a_coll

def erosion_distribution(simu,a_min,a_max,m_rem,m_tot):
    '''
    returns the particle in which the tracked particle ends up in when the fragments
    are distributed according to eq (13)
    Arguments:
    simu    = instance of the Simulation class
    a_min   = smallest collisional product, typically size of a monomer
    a_max   = largest fragment size, typically equal the size of the smaller collsion partner
    m_rem   = mass of the remaing aggregate which is being eroded
    m_tot   = total mass of all the colliding particles
    '''
    if (a_min==a_max):
        print('extra case erosion')

    m_frag = m_tot-m_rem #total mass of the fragments
    a_rem = (3./(4.*np.pi)*(m_rem/simu.disk.dust.rho_s))**(1./3.) #radius of the eroded particle
    N = 1500
    xi = 3.5
    a_coll_grid = np.logspace(np.log10(a_min), np.log10(a_rem), num=N, dtype=float)
    dloga = np.diff(np.log(a_coll_grid))[0]

    m_grid = (4./3.)*np.pi*simu.disk.dust.rho_s*a_coll_grid**3
    nf_a = np.zeros_like(a_coll_grid)
    nf_a[a_coll_grid<=a_max] = a_coll_grid[a_coll_grid<=a_max]**(-xi)

    dM_sum = np.sum(nf_a*a_coll_grid*m_grid*dloga) #total mass of the fragments in code units
    nf_a = nf_a/dM_sum*m_frag #nf_a converted to cgs units


    nf_a[-1]=1./(a_coll_grid[-1]*m_grid[-1])*m_rem/dloga #added the contribution which comes from the remaining agregate of mass m_rem

    dM = nf_a*a_coll_grid*m_grid*dloga

    Psum = np.sum(dM)
    P = 1./Psum*dM
    a_coll = np.random.choice(a_coll_grid,p=P)
    return a_coll



def erosion_distribution_ori(simu,a_min,a_max,m_rem,m_tot):
    '''
    returns the particle in which the tracked particle ends up in when the fragments
    are distributed according to eq (13)
    Arguments:
    simu    = instance of the Simulation class
    a_min   = smallest collisional product, typically size of a monomer
    a_max   = largest fragment size, typically equal the size of the smaller collsion partner
    m_rem   = mass of the remaing aggregate which is being eroded
    m_tot   = total mass of all the colliding particles
    '''
    if (a_min==a_max):
        print('extra case erosion')

    m_frag = m_tot-m_rem #total mass of all the fragments
    a_rem = (3./(4.*np.pi)*(m_rem/simu.disk.dust.rho_s))**(1./3.) #radius of the eroded particle
    N = 1500
    xi = 3.5
    a_coll_grid = np.logspace(np.log10(a_min), np.log10(a_rem), num=N, dtype=float)
    dloga = np.diff(np.log(a_coll_grid))[0]

    m_grid = (4./3.)*np.pi*simu.disk.dust.rho_s*a_coll_grid**3
    nf_a = np.zeros_like(a_coll_grid)
    nf_a[a_coll_grid<=a_max] = a_coll_grid[a_coll_grid<=a_max]**(-xi)

    dM_sum = np.sum(nf_a*a_coll_grid*m_grid*dloga) #total mass of the fragments in code units

    nf_a = nf_a/dM_sum*m_frag #nf_a converted to cgs units
    nf_a[-1]=1./(a_coll_grid[-1]*m_grid[-1])*m_rem/dloga #added the contribution which comes from the remaining agregate of mass m_rem

    dM = nf_a*a_coll_grid*m_grid*dloga

    Psum = np.sum(dM)
    P = 1./Psum*dM
    a_coll = np.random.choice(a_coll_grid,p=P)
    return a_coll


def dvt_ormel(tau_1,tau_2,t0,v0,ts,vs,Reynolds):
    """
    A function that gives the velocities according to Ormel and Cuzzi (2007), extended recipe
    Arguments
    tau_1       =   stopping time of particle 1
    tau_2       =   stopping time of particle 2
    mas_1       =   mass of particle 1
    mas_2       =   mass of particle 2
    t0          =   large eddy turnover time
    v0          =   large eddy velocity
    ts          =   small eddy turnover time
    vs          =   small eddy velocity
    Reynolds    =   Reynolds number
    RETURNS: v_rel_ormel =   relative velocity
    """
    from numpy import sqrt
    ##sort tau's 1--> correspond to the max. now
    #(a bit confusing perhaps)
    if (tau_1 >= tau_2):
        tau_mx = tau_1
        tau_mn = tau_2
        St1    = tau_mx/t0
        St2    = tau_mn/t0
    else:
        tau_mx = tau_2
        tau_mn = tau_1
        St1    = tau_mx/t0
        St2    = tau_mn/t0
    Vg2 = 1.5 *v0**2.0 # note the square
    ya  = 1.6;         # approximate solution for St*=y*St1; valid for St1 << 1.
    if (tau_mx < 0.2*ts):     #very small regime
        dv = 1.5 *(vs/ts *(tau_mx - tau_mn))**2.0 #Eq. 27
    elif (tau_mx < ts/ya):
        dv = Vg2 *(St1-St2)/(St1+St2)*(St1**2.0/(St1+Reynolds**(-0.5)) - St2**2.0/(St2+Reynolds**(-0.5))) #Eq. 26
    elif (tau_mx < 5.0*ts):
        #
        #Eq. 17 of OC07. The second term with St_i**2.0 is negligible (assuming %Re>>1)
        #hulp1 = Eq. 17; hulp2 = Eq. 18
        #
        hulp1 = ( (St1-St2)/(St1+St2) * (St1**2.0/(St1+ya*St1) - St2**2.0/(St2+ya*St1)) ) #note the -sign
        hulp2 = 2.0*(ya*St1-Reynolds**(-0.5)) + St1**2.0/(ya*St1+St1) - St1**2.0/(St1+Reynolds**(-0.5)) + St2**2.0/(ya*St1+St2) - St2**2.0/(St2+Reynolds**(-0.5))
        dv    = Vg2 *(hulp1 + hulp2)
    elif (tau_mx < t0/5.0): #"Full intermediate regime
        eps = St2/St1 #stopping time ratio
        dv  =  Vg2 *( St1*(2.0*ya - (1.0+eps) + 2.0/(1.0+eps) *(1.0/(1.0+ya) + eps**3.0/(ya+eps) )) )
    elif (tau_mx < t0):
        #now y* lies between 1.6 (St1 << 1) and 1.0 (St1>=1). The fit below fits ystar to less than 1%
        c3 =-0.29847604
        c2 = 0.32938936
        c1 =-0.63119577
        c0 = 1.6015125
        y_star = c0 + c1*St1 + c2*St1**2.0 + c3*St1**3.0
        #
        # we can then employ the same formula as before
        #
        eps = St2/St1; #stopping time ratio
        dv  = Vg2 *( St1*(2.0*y_star - (1.0+eps) + 2.0/(1.0+eps) *(1.0/(1.0+y_star) + eps**3.0/(y_star+eps) )) );
    else:
        #heavy particle limit
        dv = Vg2 *( 1.0/(1.0+St1) + 1.0/(1.0+St2) )
    return sqrt(dv)


def dv_brownian(T,rho_s,a1,a2):
    dv_bm = 0.0
    m1 = rho_s*(4.0*np.pi/3.0)*a1**3.
    m2 = rho_s*(4.0*np.pi/3.0)*a2**3.
    dv_bm = np.sqrt(8.*c.kB*T*(m1+m2)/(np.pi*m1*m2))
    return dv_bm


def v_rel_par(rho_s,rho_gz,cs,T,tau_1,a0,tn,vn,ts,vs,re,a):


    tau_2 = np.sqrt(np.pi/8.)*(rho_s*a)/(rho_gz*cs)#stopping time of particle 2
    dv_bm = dv_brownian(T=T,rho_s=rho_s,a1=a0,a2=a)
    dv_tur = dvt_ormel(tau_1,tau_2,tn,vn,ts,vs,re)

    return np.sqrt(dv_tur**2.+dv_bm**2.)


def v_rel_par2(c,b,a):
    return c*b*a
