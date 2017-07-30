'''
Nicholas Geneva
ngeneva@nd.edu
July 10, 2017
'''
import sys
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib import rc
from torch.autograd import Variable
from gptorch import densities

def plot_guassian():
    delta = 0.025
    x0 = np.arange(0, 1.1, delta)
    y0 = np.arange(0, 1.1, delta)
    X, Y = np.meshgrid(x0, y0)
    P = np.zeros((len(x0),len(y0)))
    P1 = np.zeros(len(x0)) #P(a|b)
    P2 = np.zeros(len(x0)) #P(a)

    #Calculate Gaussian in 2 dimensions
    mu = th.FloatTensor([[0.5],[0.5]])
    covar = th.FloatTensor([[0.0115, 0.01], [0.01, 0.012]])
    for i in range(0,len(x0)):
        for j in range(0,len(y0)):
            x = th.FloatTensor([[x0[i]],[y0[j]]])
            p = densities.gaussianMD(x,mu,covar)
            P[i,j] = Variable(p).data.numpy()

    #Calculate specific profiles
    xb = 0.7
    for i in range(0,len(x0)): #Calculate specific P(a|b) profile
        x = th.FloatTensor([[x0[i]],[xb]])
        p = densities.gaussianMD(x,mu,covar)
        P1[i] = Variable(p).data.numpy()

    mu = th.FloatTensor([0.5]) #mu_aa
    covar = th.FloatTensor([0.0115]) #sigma_aa
    for i in range(0,len(x0)): #Calculate specific P(a) profile
            x = th.FloatTensor([x0[i]])
            p = densities.gaussian1D(x,mu,covar)
            P2[i] = Variable(p).data.numpy()

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.suptitle('Figure 2.9, pg. 90', fontsize=14)

    #Contour Plot of Guassian
    CS = ax1.contour(X, Y, P, 5)
    ax1.clabel(CS, inline=1, fontsize=10)
    line1, = ax1.plot([0, 1], [xb, xb], '-k',label=r'$x_b = 0.7$')
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])
    ax1.set_xlabel(r'$x_a$')
    ax1.set_ylabel(r'$x_b$')
    ax1.legend(handles=[line1], loc=2)
    ax1.text(0.6, 0.3, r'$p(x_a , x_b)$')
    
    #Profile plots
    line1, = ax2.plot(x0,P1,'-r',label=r'$p(x_a | x_b = 0.7)$')
    line2, = ax2.plot(x0,P2,'-b',label=r'$p(x_a)$')
    ax2.set_xlim([0,1])
    ax2.set_xlabel(r'$x_a$')
    ax2.legend(handles=[line1, line2], loc=2)

    #plt.savefig('Figure2_9.png')
    plt.show()


if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    plot_guassian()
