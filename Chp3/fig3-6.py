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

    return [X, T]

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    
    L = 100 #Number of data sets
    M = 25 #Number of basis
    P = 25 #Number of points to calculate the solved regression
    X_train,T_train = generateData(L,P,0.3) #Generate training data
    X_test,T_test = generateData(1,1000,0.3) #Generate test data

    gammas = np.linspace(-2.5,1.5,50)
    Y0 = th.DoubleTensor(len(gammas),P,L).zero_()
    Y1 = th.DoubleTensor(len(gammas),1000,L).zero_()

    for idx, val in enumerate(gammas):
        lsr = LeastSquaresReg(M, np.exp(val), basis='guassian')
        lsr.calcRegression(X_train,T_train)
        Y0[idx,:,:] = th.mm(lsr.getBasis(X_train), lsr.getWeights().squeeze())
        Y1[idx,:,:] = th.mm(lsr.getBasis(X_test), lsr.getWeights())

    Y0_avg = th.squeeze(th.sum(Y0,2))/L #Average over all training sets
    X, H = generateData(1,P,0)
    H = H.t().expand(len(gammas), P)

    #For some reason BIAS and VAR error seem to be off by a factor of 2???
    BIAS = th.sum(th.pow(Y0_avg - H,2),1)/(2*P) #Eq. 3.46
    VAR = th.pow(Y0_avg.unsqueeze(2).expand(Y0.size()) - Y0, 2)
    VAR = th.squeeze(th.sum(th.sum(VAR, 1), 2)/(L*P)) #Eq. 3.47
    #Calculate test set data error for a set with 1000 points
    X_test,T_test = generateData(1,1000,0) #Generate test data
    TERR = th.pow(Y1 - T_test.unsqueeze(0).expand(Y1.size()),2)
    TERR = th.squeeze(th.sum(th.sum(TERR,1),2))/(1000*L)

    # Two subplots, unpack the axes array immediately
    plt.figure(figsize=(8, 5))
    plt.suptitle('Figure 3.6, pg. 151', fontsize=14)
    line1, = plt.plot(gammas, Variable(BIAS).data.numpy(), '-r',label=r'$\left( bias \right)^2$')
    line2, = plt.plot(gammas, Variable(VAR).data.numpy(), '-b',label=r'$variance$')
    line3, = plt.plot(gammas, Variable(BIAS+VAR).data.numpy(), '-m',label=r'$\left( bias \right)^2 + variance$')
    line4, = plt.plot(gammas, Variable(TERR).data.numpy(), '-k',label=r'$test ~error$')

    plt.xlim([-3,2])
    plt.ylim([0,0.15])
    plt.xlabel(r'$ln(\lambda)$')
    plt.legend(handles=[line1, line2, line3, line4], loc=2)
    #plt.savefig('Figure3_6.png')
    plt.show()
