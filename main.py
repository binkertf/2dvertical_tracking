
import numpy as np
import matplotlib.pyplot as plt
from simu import Simu
import constants as c

import sys


def zdata(t_tot=20000., a=1.0):
    simulation = Simu()
    simulation.particle.a=a
    simulation.run(T)
    hg = simulation.disk.gas.h #gas scale height
    return simulation.parameters.data[:,1]/hg, simulation.parameters.data[:,0]


def histogram(t_tot = 20000., N=20,a=1.0,norm=True):
    bins = 100
    binrange = (-3.5,3.5)
    data, weights = zdata(T,a)
    H=np.histogram(data,bins=bins,range=binrange,weights=weights)

    for i in range(N-1):
        data, weights = zdata(T,a)
        H2 = np.histogram(data,bins=bins,range=binrange,weights=weights)
        H = (H[0]+H2[0],H[1])

        #print a status bar
        sys.stdout.write('\riter ' + str(i) + ' of ' + str(N-1))
        sys.stdout.flush()
    if norm == True:
        nrm = np.sum(H[0])*((binrange[1]-binrange[0])/bins)
        H = (H[0]/nrm,H[1])

    return H



##############################################################################

a = ([1.0,0.1,0.0001])
label = [r'$1\:cm$', r'$0.1\:cm$', r'$1\:\mu m$']


'''
fig=plt.figure()
for i in range(len(a)):
    test_simu=Simu()
    test_simu.particle.a=a[i]
    #test_simu.particle.z=0.0
    test_simu.run(T=20000.)
    Z1 = test_simu.parameters.data
    hg = test_simu.disk.gas.h
    plt.plot(Z1[:,0]/c.yr,Z1[:,1]/hg,label=label[i])
    print(Z1.shape)


plt.xscale('linear')
plt.title('vertical settling, [Krijt (2016)]')
plt.xlabel('time [yr]')
plt.ylabel(r'$z/h_g$')
plt.ylim((-3.5,3.5))
plt.xlim((0,20000))
plt.legend()
plt.grid()
plt.show()
#plt.savefig('vertical_settling.png')
'''


#histogram

plt.figure()
for i in range(len(a)):
    hist = histogram(T = 20000., N=20,a=a[i],norm=True)
    #plt.step(hist[1][:-1],hist[0]/np.amax(hist[0]),where='post')
    plt.step(hist[1][:-1],hist[0],where='post')


plt.yscale('log')
plt.xlim((-3.5,3.5))
plt.ylim((1.0e-3,4))
plt.ylabel('normalized integrated time')
plt.xlabel(r'$z/h_g$')
plt.savefig('histogram.png')
plt.show()




#test_simu.plot_dust_size_distribution()
#test_simu.check_dust_sigma()
