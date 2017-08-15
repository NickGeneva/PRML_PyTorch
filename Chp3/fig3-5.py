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

class LeastSquaresReg():
    def __init__(self, m, lmbda, basis='poly'):
        '''
        Least squares class
        Args:
            m (Variable) = number of basis functions
            lmbda (Variable) = regularization parameter
            basis (Variable) = type of basis function
        '''
        self.m = m
        self.lmbda = lmbda
        #Set up weight and mu arrays
        if(m > 2):
            self.mu = th.linspace(0,1,self.m-1).unsqueeze(1)
            self.w0 = th.linspace(0,1,self.m).unsqueeze(1)
        elif(m > 1):
            self.mu = th.FloatTensor([[0.5]])
            self.w0 = th.linspace(0,1,self.m).unsqueeze(1)
        else:
            self.mu = 0
            self.w0 = th.FloatTensor([[0]])
        
        self.basis = basis

    def calcRegression(self, X, T):
        '''
        Calculates the weights of the linear model using minimization of least squares
        Args:
            X (th.DoubleTensor) = Nx1 column vector of target points X-coords
            T (th.DoubleTensor) = NxL matrix of targer values with L different data sets
        '''
        phi = self.getBasis(X)
        #Eq. 3.28
        w = th.mm(th.transpose(phi,0,1), T.type(th.DoubleTensor)) #NxM matrix (multi-output approach)
        w2 = th.mm(th.transpose(phi,0,1), phi) #MxM matrix
        w2 = th.inverse(self.lmbda*th.eye(self.m).type(th.DoubleTensor) + w2)
        self.w0 = th.mm(w2,w)

    def getBasis(self, X):
        '''
        Generates basis matrix, current supported basis functions are polynomial and guassian
        Args:
            X (th.DoubleTensor) = Nx1 column vector of target points X-coords
        Returns:
            phi (th.DoubleTensor) = NxM matrix of basis functions for each point
        '''
        if(self.basis == 'poly'): #Polynomial basis
            if(self.m > 1):
                exp = th.linspace(0,self.m-1,self.m).unsqueeze(0).expand(X.size(0),self.m).type(th.DoubleTensor)
                phi = th.pow(X.expand(X.size(0),self.m), exp)
            else:
                exp = th.DoubleTensor([[0]])
                phi = th.pow(X.expand(X.size(0),self.m),0)
        else: #Guassian basis
            s = 1.0/(self.m-1)
            mu = th.transpose(self.mu,0,1).type(th.DoubleTensor)
            phi = th.DoubleTensor(X.size(0),self.m).zero_() + 1
    
            phi0 = th.pow(X.expand(X.size(0),self.m-1)-mu.expand(X.size(0),self.m-1),2)
            phi[:,1::] = th.exp(-phi0/(2.0*s**2))
        
        return phi

    def getWeights(self):
        '''
        Get regression weights
        Returns:
            phi (th.DoubleTensor) = NxM matrix of basis functions for each point
        '''
        return self.w0

    def getTestError(self, X, T):
        '''
        Calculate RMS test error
        Args:
            X (th.DoubleTensor) = Nx1 column vector of test points X-coords
            T (th.DoubleTensor) = Nx1 matrix of test values with L different data sets
        Returns:
            err (Variable) = RMS error 
        '''
        N = T.size(0)
        phi = th.mm(self.getBasis(X),self.w0)
        err = th.sqrt(th.sum(th.pow(T - phi,2), 0)/N) #Eq. 1.3
        return Variable(err).data.numpy()
        
def generateData(L,N,std):
    """Generates a set of synthetic data evenly distributed along the X axis
        with a target function of sin(2*pi*x) with guassian noise in the Y direction
    Args:
        L (Variable) = Number of data sets desired
        N (Variable) = Number of points in data set
        std (Array) = standard deviation of guassian noise
    Returns:
        X (th.DoubleTensor) = NxL matrix of X coords of target points
        T (th.DoubleTensor) = NxL matrix of Y coord of target points
    """
    X = th.linspace(0,1,N).unsqueeze(1).type(th.DoubleTensor)
    mu = th.sin(2.0*np.pi*X).expand(N,L)
    if(std > 0):
        T = th.normal(mu,std).type(th.DoubleTensor)
    else:
        T = mu.type(th.DoubleTensor)

    #Plot a sample data set
    #plotGeneratedData(X0,Y0,Variable(X1).data.numpy(),Variable(T[:,0]).data.numpy(), std)

    return [X, T]

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

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    # Two subplots, unpack the axes array immediately
    f, ax = plt.subplots(3, 2, figsize=(8, 7))
    f.suptitle('Figure 3.5, pg. 150', fontsize=14)

    X_train,T_train = generateData(100,25,0.3)
    X_test,T_test = generateData(1,100,0)
    X0 = Variable(X_test).data.numpy()
    M = 25
    lam = [2.6, -0.31, -2.4] #ln(lambda) values

    for idx, val in enumerate(lam):
        lsr = LeastSquaresReg(M, np.exp(val), basis='guassian')
        lsr.calcRegression(X_train, T_train)
        Y_fit = th.mm(lsr.getBasis(X_test), lsr.getWeights())

        for i in range(0,20): #Plot 20 of the regressions
            ax[idx,0].plot(X0, Variable(Y_fit[:,i]).data.numpy(), '-r', lw=0.2)
        ax[idx,0].text(0.6, 1, r'$ln \lambda = 2.6$')
        ax[idx,1].plot(X0, np.sin(2.0*np.pi*X0),'-g')
        ax[idx,1].plot(X0, Variable(th.sum(Y_fit,1)/100).data.numpy(),'-r')

    for (m,n), subplot in np.ndenumerate(ax):
        ax[m,n].set_xlim([0,1])
        ax[m,n].set_ylim([-1.5,1.5])
        ax[m,n].set_xlabel(r'$x$')
        ax[m,n].set_ylabel(r'$t$',rotation=0)
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5, rect=[0,0, 1, 0.9])
    #plt.savefig('Figure3_05.png')
    plt.show()
