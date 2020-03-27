import constants as c
import numpy as np
from numpy import sqrt
from scipy import integrate


def temperature(simu):
    '''
    computes the local temperature at radius R based on Krijt (2016)
    '''

    return 280.*(simu.disk.R/c.AU)**(-0.5)

def background_disrtibution(simu):

    a_grid1 = simu.disk.dust.a_grid
    sig = np.ones_like(a_grid1)
    sig[a_grid1>2.0e-2] = 0.0
    dloga = np.diff(np.log(a_grid1))[0]

    C = np.sum(sig*dloga)
    sig = sig*simu.disk.dust.sig/C
    return sig

def background_disrtibution2(simu):

    sig = np.array([7.09623679e-02, 3.96576864e-02, 3.45393367e-02, 7.35501778e-02,
       8.30740646e-02, 6.90483048e-02, 8.80992842e-02, 7.40558343e-02,
       9.52244269e-02, 5.81021853e-02, 1.25488574e-01, 9.63149259e-02,
       9.77796401e-02, 1.03622630e-01, 1.06714044e-01, 1.09185780e-01,
       1.10941888e-01, 1.16256879e-01, 1.44687940e-01, 1.04709352e-01,
       1.36555719e-01, 1.39453599e-01, 1.61227588e-01, 1.21336532e-01,
       1.13530403e-01, 1.36406321e-01, 1.41849863e-01, 1.47275781e-01,
       1.55308221e-01, 1.23991474e-01, 1.62812808e-01, 1.45114414e-01,
       1.49150545e-01, 1.06577179e-01, 1.38105964e-01, 1.65269668e-01,
       1.66139876e-01, 1.80752316e-01, 1.34779819e-01, 1.54364741e-01,
       1.70062875e-01, 1.56951021e-01, 1.50574141e-01, 1.02636713e-01,
       1.28851113e-01, 1.17818243e-01, 1.41381753e-01, 1.65605913e-01,
       1.63520991e-01, 1.37499701e-01, 1.29907796e-01, 1.71655383e-01,
       1.71292301e-01, 1.32270975e-01, 1.75991259e-01, 1.64747589e-01,
       1.68514211e-01, 2.04770579e-01, 1.27484489e-01, 1.60537272e-01,
       1.92036867e-01, 1.80460136e-01, 1.61649252e-01, 1.73640936e-01,
       1.73075660e-01, 1.55244090e-01, 1.77201230e-01, 1.99111947e-01,
       2.32530422e-01, 2.37050935e-01, 2.56298695e-01, 2.03692923e-01,
       2.51166323e-01, 2.16000697e-01, 2.29520053e-01, 2.57824501e-01,
       2.25211868e-01, 2.29797369e-01, 2.50494406e-01, 2.07002399e-01,
       2.52257595e-01, 2.47719166e-01, 2.41628753e-01, 2.79580154e-01,
       2.08352784e-01, 2.69042695e-01, 2.57111308e-01, 2.94463742e-01,
       2.62255248e-01, 2.31831015e-01, 2.35463679e-01, 2.29560312e-01,
       2.53465332e-01, 2.53595570e-01, 2.62361677e-01, 2.70657199e-01,
       4.12050508e-01, 3.06377336e-01, 3.29085090e-01, 3.55896927e-01,
       3.09953745e-01, 3.41956731e-01, 3.81913814e-01, 3.33319719e-01,
       3.44186866e-01, 3.95156913e-01, 3.61339221e-01, 3.27445333e-01,
       3.58631726e-01, 3.53047735e-01, 3.17572228e-01, 3.98645981e-01,
       3.87539440e-01, 4.03180882e-01, 3.58913179e-01, 3.80679780e-01,
       4.12399915e-01, 3.44581214e-01, 3.41217014e-01, 3.18343170e-01,
       3.15261275e-01, 3.73183618e-01, 3.43570046e-01, 3.45275834e-01,
       3.51974566e-01, 3.60227841e-01, 4.30844614e-01, 4.49136226e-01,
       4.92631181e-01, 5.02390352e-01, 4.82971102e-01, 5.26955399e-01,
       5.42349662e-01, 5.60931414e-01, 6.33335577e-01, 6.42888259e-01,
       6.45243027e-01, 6.27704779e-01, 7.09504362e-01, 6.94335657e-01,
       7.26167213e-01, 7.19816177e-01, 7.77113155e-01, 8.23995603e-01,
       8.00738458e-01, 8.08678859e-01, 8.47898318e-01, 9.04410460e-01,
       8.87957850e-01, 9.21263835e-01, 9.92563283e-01, 9.70035067e-01,
       9.84969510e-01, 1.12751596e+00, 1.08795681e+00, 1.17129493e+00,
       1.20337687e+00, 1.20473051e+00, 1.21081055e+00, 1.24610156e+00,
       1.37688588e+00, 1.40070190e+00, 1.46199414e+00, 1.44677330e+00,
       1.53305105e+00, 1.52622899e+00, 1.57831429e+00, 1.66680187e+00,
       1.68897558e+00, 1.76990133e+00, 1.74921212e+00, 1.85684739e+00,
       1.96438460e+00, 1.98903375e+00, 1.96646966e+00, 1.97578802e+00,
       2.12666328e+00, 2.11698627e+00, 2.21510474e+00, 2.33201315e+00,
       2.39340385e+00, 2.48426151e+00, 2.57268093e+00, 2.57803599e+00,
       2.70506802e+00, 2.77136804e+00, 2.86087123e+00, 2.94231661e+00,
       3.01272478e+00, 3.14932353e+00, 3.25647352e+00, 3.31889414e+00,
       3.49934304e+00, 3.58004389e+00, 3.50954102e+00, 3.74435387e+00,
       3.92124198e+00, 4.18553813e+00, 4.44464597e+00, 4.70920124e+00,
       5.08622153e+00, 5.67208429e+00, 6.26953121e+00, 7.16063343e+00,
       7.97419946e+00, 9.05458170e+00, 1.02191339e+01, 1.07369213e+01,
       1.03852040e+01, 9.37260080e+00, 7.99683578e+00, 6.50803129e+00,
       5.18792703e+00, 3.85340643e+00, 2.57113419e+00, 1.65806827e+00,
       9.36425221e-01, 4.71343258e-01, 2.09371488e-01, 7.67024789e-02,
       1.23599699e-02, 1.42640847e-03, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    a_grid1 = simu.disk.dust.a_grid


    a_grid1 = simu.disk.dust.a_grid
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
