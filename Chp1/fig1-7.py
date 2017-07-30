'''
Nicholas Geneva
ngeneva@nd.edu
July 23, 2017
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
            mu = th.transpose(self.mu,0,1)
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

    return [X, T]

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    f, ax = plt.subplots(1, 2, figsize=(9, 4))
    f.suptitle('Figure 1.7, pg. 10', fontsize=14)

    X_train, T_train = generateData(1, 10, 0.3)
    X_test, T_test = generateData(1, 100, 0)


    X0 = np.linspace(0, 1, 100)
    lmbda = np.array([-18, 0])
    
    for n, ax0 in enumerate(ax):
        lsr = LeastSquaresReg(10,np.exp(lmbda[n]))
        lsr.calcRegression(X_train,T_train)
        
        X_tensor = th.DoubleTensor(X0).unsqueeze(1)
        Y0 = Variable(th.mm(lsr.getBasis(X_tensor), lsr.getWeights())).data.numpy()

        ax0.plot(Variable(X_test).data.numpy(), Variable(T_test).data.numpy(), '-g')
        ax0.plot(Variable(X_train).data.numpy(), Variable(T_train).data.numpy(), 'o', markersize=7, \
        markeredgewidth=1,markeredgecolor='b', \
          markerfacecolor='None')
        ax0.plot(X0, Y0, '-r')

        ax0.text(0.7, 1.0, r'$ln ~\lambda='+str(lmbda[n])+'$')
        ax0.set_xlim([-0.1, 1.1])
        ax0.set_ylim([-1.5, 1.5])
        ax0.set_xlabel(r'$x$')
        ax0.set_ylabel(r'$t$')
    
    plt.tight_layout(rect=[0,0, 1.0, 0.93])
    #plt.savefig('Figure1_7.png')
    plt.show()
