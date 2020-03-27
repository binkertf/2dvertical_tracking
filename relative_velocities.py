'''
this is a section from Til python script based on Ormels Fortran script. It is based on Ormel & Cuzzi (2007)
'''

re    = CONST_ALPHA*sig_h2*sigma_g[it,ir]/(2.*mu*m_p)
            rs    = RHO_S                               #dust solid density
            c     = sqrt(k_b*T[it,ir]/mu/m_p)           #isothermal sound speed
            omega = sqrt(Grav*m_star[it]/x[ir]**3)      #keplerian frquency
            tn    = 1./omega                            #overturn time of the larges eddies (1 orbital timescale)
            ts    = tn*re**(-0.5)                       #overturn time of the smallest eddies
            vn    = sqrt(CONST_ALPHA)*c                 #turbulent velocity dispersion in gas
            vs    = vn*re**(-0.25)                      #velocity of the smallest eddie (?)
            tau_1 = RHO_S*grainsizes[mi]/sigma_g[it,ir]/omega*pi/2. #stopping time of particel 1 (at the midplane)
            tau_2 = RHO_S*grainsizes[mj]/sigma_g[it,ir]/omega*pi/2. #stopping time of particle 2 (at the midplane)
            dv_TM[i,j] = dvt_ormel(tau_1,tau_2,tn,vn,ts,vs,re)


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
        dv = 1.5 *(vs/ts *(tau_mx - tau_mn))**2.0
    elif (tau_mx < ts/ya):
        dv = Vg2 *(St1-St2)/(St1+St2)*(St1**2.0/(St1+Reynolds**(-0.5)) - St2**2.0/(St2+Reynolds**(-0.5)))
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

################################################################################

################################################################################
