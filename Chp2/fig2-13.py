'''
Nicholas Geneva
ngeneva@nd.edu
July 11, 2017
'''
import sys
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib import rc
from torch.autograd import Variable
from GPtorch import densities

def plot_gamma():
    delta = 0.025
    x0 = np.arange(0.01, 2.1, delta)
    P1 = np.zeros(len(x0)) # a=0.1, b=0.1
    P2 = np.zeros(len(x0)) # a=1,b=1
    P3 = np.zeros(len(x0)) # a=4,b=6

    for i in range(0,len(x0)): #Calculate specific P(a) profile
            P1[i] = densities.gamma1D(x0[i],0.1,0.1)
            P2[i] = densities.gamma1D(x0[i],1,1)
            P3[i] = densities.gamma1D(x0[i],4,6)

    # Two subplots, unpack the axes array immediately
    f, ax = plt.subplots(1, 3, figsize=(8, 3))
    f.suptitle('Figure 2.13, pg. 100', fontsize=14)
    
    #Profile plots
    #line1, = ax2.plot(x0,P1,'-r',label=r'$p(x_a | x_b = 0.7)$')
    ax[0].plot(x0,P1,'-r')
    ax[0].set_xlim([0,2])
    ax[0].set_ylim([0,2])
    ax[0].set_xlabel(r'$\lambda')
    ax[0].text(0.8, 1.7, r'$a=0.1, b=0.1$')

    ax[1].plot(x0,P2,'-r')
    ax[1].set_xlim([0,2])
    ax[1].set_ylim([0,2])
    ax[1].set_xlabel(r'$\lambda$')
    ax[1].text(0.8, 1.7, r'$a=1.0, b=1.0$')

    ax[2].plot(x0,P3,'-r')
    ax[2].set_xlim([0,2])
    ax[2].set_ylim([0,2])
    ax[2].set_xlabel(r'$\lambda$')
    ax[2].text(0.8, 1.7, r'$a=4.0, b=6.0$')

    plt.tight_layout(rect=[0,0, 1, 0.93])
    #plt.savefig('Figure2_13.png')
    plt.show()


if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    plot_gamma()
