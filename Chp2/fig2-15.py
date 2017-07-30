'''
Nicholas Geneva
ngeneva@nd.edu
July 12, 2017
'''
import sys

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib import rc
from torch.autograd import Variable

#Tell python were else to look for packages
sys.path.insert(0,"/home/nick/Documents/GPtorch/")
from GPtorch import densities


def plot_st():
    delta = 0.1
    x0 = np.arange(-5, 5.01, delta)
    P1 = np.zeros(len(x0)) # a=0.1, b=0.1
    P2 = np.zeros(len(x0)) # a=1,b=1
    P3 = np.zeros(len(x0)) # a=4,b=6

    mu = th.FloatTensor([[0]])
    lam = th.FloatTensor([[1.0]])
    for i in range(0,len(x0)): #Calculate specific P(a) profile
            x = th.FloatTensor([[x0[i]]])
            p = densities.studentT(x,mu,lam,0.1)
            P1[i] = Variable(p).data.numpy()
            p = densities.studentT(x,mu,lam,1.0)
            P2[i] = Variable(p).data.numpy()
            p = densities.studentT(x,mu,lam,1e5)
            P3[i] = Variable(p).data.numpy()

    # Two subplots, unpack the axes array immediately
    f, ax = plt.subplots(1, 1)
    f.suptitle('Figure 2.15, pg. 103', fontsize=14)
    
    #Profile plots
    #line1, = ax2.plot(x0,P1,'-r',label=r'$p(x_a | x_b = 0.7)$')
    line1, = ax.plot(x0,P1,'-r',label=r'$\nu = 0.1$')
    line2, = ax.plot(x0,P2,'-b',label=r'$\nu = 1.0$')
    line3, = ax.plot(x0,P3,'-g',label=r'$\nu \rightarrow \infty$')
    ax.set_xlim([-5,5])
    ax.set_ylim([0,0.5])
    ax.set_xlabel(r'$x')
    ax.legend(handles=[line3, line2, line1], loc=1)

    #plt.savefig('Figure2_15.png')
    plt.show()


if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    plot_st()
