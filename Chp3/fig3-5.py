'''
Nicholas Geneva
ngeneva@nd.edu
July 14, 2017
'''
import sys

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib import rc
from torch.autograd import Variable

def generateData(L,N,std):
    """Generates a set of synthetic data evenly distributed along the X axis
        with a target function of sin(2*pi*x) with guassian noise in the Y direction
    Args:
        L (Variable) = Number of data sets desired
        N (Variable) = Number of points in data set
        std (Array) = standard deviation of guassian noise
    Returns:
        X (th.FloatTensor) = NxL matrix of X coords of target points
        T (th.FloatTensor) = NxL matrix of Y coord of target points
    """
    step = 1.0/(N-1)
    X1 = th.arange(0,1,step).expand(1,N)
    X1 = th.transpose(X1,0,1)
    mu = th.sin(2.0*np.pi*X1).expand(N,L)
    T = th.normal(mu,std)
    #Plot a sample data set
    #plotGeneratedData(X0,Y0,Variable(X1).data.numpy(),Variable(T[:,0]).data.numpy(), std)

    return [X1, T]

def plotGeneratedData(X0,Y0,X1,Y1,std):
    """Plot generated random data to be used for regression like Figure A.6, pg. 683 in PRML
  
      Args:
          X0 (Array) = Mx1 column of target function X
          Y0 (Array) = Mx1 column of target function Y
          X1 (Array) = Nx1 column of data points X values
          Y1 (Array) = Nx1 column of data points Y values with guassian noise
          std (Variable) = Standard deviation of the added guassian noise 
      """
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.suptitle('Figure A.6, pg. 683', fontsize=14)

    #Plot data points versus target function
    line1, = ax1.plot(X0, Y0, '-g',label=r'$sin(2\pi x)$')
    line2, = ax1.plot(X1, Y1, 'o', markersize=7, \
        markeredgewidth=1,markeredgecolor='b', \
          markerfacecolor='None', label=r'Synthetic Data')
    ax1.set_xlim([0,1])
    ax1.set_ylim([-1.5,1.5])
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$t$')
    ax1.legend(handles=[line1,line2], loc=1)

    #Plot target function and +/- std range
    line1, = ax2.plot(X0, Y0, '-g',label=r'$\mu$')
    lsigma = ax2.fill_between(X0, Y0-std, Y0+std, facecolor='r',alpha=0.5,label=r'$\pm \sigma$')
    ax2.set_xlim([0,1])
    ax2.set_ylim([-1.5,1.5])
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$t$')
    ax2.legend(handles=[line1,lsigma], loc=1)

    plt.show()

def regression(X,T,M,P,lamb):
    '''Calculates multi-output regression with guassian basis, assumes all data sets has
        the same X-coordinates.
  
      Args:
          X (th.FloatTensor) = Nx1 column vector of target points X-coords
          T (th.FloatTensor) = NxL matrix of targer values with L different data sets
          M (Variable) = Number of regression basis (includes phi0)
          P (Variable) = Number of points to calculate the solved regression function with
          lamb (Variable) = regularization term to control impact of regularization error
    Returns:
        X0 (th.FloatTensor) = X-coord of regressions
        Y (th.FloatTensor) = PxL matrix of regression points for each data set
    '''
    N = T.size(0) #Number of data points in sample
    L = T.size(1) #Number of samples

    #Set up basis functions
    mu = th.FloatTensor(1,M-1).zero_()
    mu[0,:] = th.linspace(0,1,M-1)
    s = 1.0/M #scale factor
    phi = th.FloatTensor(N,M).zero_() + 1
    phi[:,1::] = th.exp(-th.pow(X.expand(N,M-1)-mu.expand(N,M-1),2)/(2*s**2))
   
    #Determine regression basis weights
    w = th.mm(th.transpose(phi,0,1),T) #NxM matrix (multi-output approach)
    w2 = th.mm(th.transpose(phi,0,1),phi) #MxM matrix
    w2 = th.inverse(lamb*th.eye(M,M) + w2)
    w = th.mm(w2,w)

    #calculate regression points
    X0 = th.FloatTensor(P,1).zero_()
    X0[:,0] = th.linspace(0,1,P)
    phi = th.FloatTensor(P,M).zero_() + 1
    phi[:,1::] = th.exp(-th.pow(X0.expand(P,M-1)-mu.expand(P,M-1),2)/(2*s**2))
    
    return [X0, th.mm(phi,w)]


if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    X,T = generateData(100,25,0.3)
    X0,Y0 = regression(X,T,25,100,np.exp(2.6))
    X1,Y1 = regression(X,T,25,100,np.exp(-0.31))
    X2,Y2 = regression(X,T,25,100,np.exp(-2.4))

    # Two subplots, unpack the axes array immediately
    f, ax = plt.subplots(3, 2)
    f.suptitle('Figure 3.5, pg. 150', fontsize=14)

    x = Variable(X0).data.numpy()
    #lambda 1
    for i in range(0,20): #Plot 20 of the regressions
        ax[0,0].plot(Variable(X0).data.numpy(), Variable(Y0[:,i]).data.numpy(), '-r', lw=0.2)
    ax[0,0].text(0.6, 1, r'$ln \lambda = 2.6$')
    ax[0,1].plot(x, np.sin(2.0*np.pi*x),'-g')
    ax[0,1].plot(x, Variable(th.sum(Y0,1)/100).data.numpy(),'-r')

    #lambda 2
    for i in range(0,20):
        ax[1,0].plot(Variable(X1).data.numpy(), Variable(Y1[:,i]).data.numpy(), '-r', lw=0.2)
    ax[1,0].text(0.6, 1, r'$ln \lambda = -0.31$')
    ax[1,1].plot(x, np.sin(2.0*np.pi*x),'-g')
    ax[1,1].plot(x, Variable(th.sum(Y1,1)/100).data.numpy(),'-r')

    #lambda 3
    for i in range(0,20):
        ax[2,0].plot(Variable(X2).data.numpy(), Variable(Y2[:,i]).data.numpy(), '-r', lw=0.2)
    ax[2,0].text(0.6, 1, r'$ln \lambda = -2.4$')
    ax[2,1].plot(x, np.sin(2.0*np.pi*x),'-g')
    ax[2,1].plot(x, Variable(th.sum(Y2,1)/100).data.numpy(),'-r')

    for (m,n), subplot in np.ndenumerate(ax):
        ax[m,n].set_xlim([0,1])
        ax[m,n].set_ylim([-1.5,1.5])
        ax[m,n].set_xlabel(r'$x$')
        ax[m,n].set_ylabel(r'$t$',rotation=0)
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5, rect=[0,0, 1, 0.9])
    #plt.savefig('Figure3_5.png')
    plt.show()
