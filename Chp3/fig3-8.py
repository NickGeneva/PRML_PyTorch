'''
Nicholas Geneva
ngeneva@nd.edu
July 19, 2017
'''
import sys

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib import rc
from torch.autograd import Variable

class BayesianLinearReg():
    def __init__(self, m, m0, alpha, beta, basis='poly'):
        '''
        Bayesian Linear Regression class
        Args:
            m (Variable) = Number of basis functions
            m0 (th.DoubleTensor) = Mx1 prior means
            alpha (Variable) = Inverse of covariance constant 
            beta (Variable) = Noise precision parameter
            basis (string) = Type of basis function
        '''
        self.m = m
        self.m0 = m0
        self.s0 = (1/alpha)*th.eye(m).type(th.DoubleTensor)
        self.alpha = alpha
        self.beta = beta
        #Set up mu arrays for guassian basis
        if(m > 2):
            self.mu = th.linspace(0,1,self.m-1).unsqueeze(1)
        elif(m > 1):
            self.mu = th.DoubleTensor([[0.5]])
        else:
            self.mu = 0
        self.basis = basis
    
    def posterUpdate(self, X, T):
        '''
        Calculates the weights of the linear model using bayesian approach
        Args:
            X (th.DoubleTensor) = Nx1 column vector of target points X-coords
            T (th.DoubleTensor) = NxL matrix of targer values with L different data sets
        '''
        phi = self.getBasis(X)
        #Eq. 3.51
        sn_i = th.inverse(self.s0) + self.beta*th.mm(th.transpose(phi,0,1),phi)
        sn = th.inverse(sn_i)
        #Eq. 3.50
        mn = th.mm(th.inverse(self.s0),self.m0) + self.beta*th.mm(th.transpose(phi,0,1),T)
        mn = th.mm(sn,mn)

        #Our updated mean and covariance is now our prior
        self.m0 = mn
        self.s0 = sn

    def getBasis(self, X):
        '''
        Generates basis matrix, current supported basis functions are polynomial and guassian
        Args:
            X (th.DoubleTensor) = Nx1 column vector of target points X-coords
        Returns:
            phi (th.DoubleTensor) = NxM matrix of basis functions for each point
        '''
        if(self.basis == 'poly'): #Polynomial basis
            exp = th.linspace(0,self.m-1,self.m).unsqueeze(0).expand(X.size(0),self.m).type(th.DoubleTensor)
            phi = th.pow(X.expand(X.size(0),self.m), exp)
        else: #Guassian basis
            s = 1.0/(self.m-1)
            mu = th.transpose(self.mu,0,1).type(th.DoubleTensor)
            phi = th.DoubleTensor(X.size(0),self.m).zero_() + 1
    
            phi0 = th.pow(X.expand(X.size(0),self.m-1)-mu.expand(X.size(0),self.m-1),2)
            phi[:,1::] = th.exp(-phi0/(2.0*s**2))
        
        return phi

    def getWeights(self):
        '''
        Generate weights for regression function probabilistically
        Returns:
            w (th.DoubleTensor) = MxL matrix of basis functions for each point
        '''
        mu = Variable(self.m0.squeeze()).data.numpy()
        covar =  Variable(self.s0.squeeze()).data.numpy()
        norm = np.random.multivariate_normal(mu, covar)
        return th.DoubleTensor(norm).unsqueeze(1)

    def getWeightsMAP(self):
        '''
        Get maximum probabilistic regression weights (i.e. means)
        Returns:
            m0 (th.DoubleTensor) = MxL matrix of basis functions for each point
        '''
        return self.m0

    def getKernel(self, X, X_n):
        '''
        Calculate the kernel of the current regression model
        Args:
            X (th.DoubleTensor) = Nx1 column vector of desired X points to predict
            X_n (th.DoubleTensor) = Px1 column vector of target points X-coords
        Returns:
            k (th.DoubleTensor) = PxN smoother matrix of equivalent kernel
        '''
        phi = self.getBasis(X)
        phi_n = self.getBasis(X_n)
        #Eq. 3.62
        k = th.mm(self.s0, th.transpose(phi,0,1))
        return self.beta*th.mm(phi_n, k)

    def getPrior(self, W):
        '''
        Gets the Prior distrubtion
        Args:
            W (th.DoubleTensor) = Mx1 column vector of input weight values
        Returns:
            P (th.DoubleTensor) = Probability of the given weights
        '''
        #Eq. 3.10
        P = th.mm(th.inverse(self.s0), (W-self.m0))
        P = th.mm(th.transpose(W-self.m0,0,1), P)
        P = th.exp(-0.5*P)
        return P

    def getLikelihood(self, W, X, T):
        '''
        Gets the likelihood distrubtion given new training data
        Args:
            W (th.DoubleTensor) = Mx1 column vector of input weight values
            X (th.DoubleTensor) = Nx1 column vector of new X training data
            T (th.DoubleTensor) = Nx1 column vector of new Y training data
        Returns:
            P (th.DoubleTensor) = Probability of the given weights
        '''
        phi = th.mm(self.getBasis(X),W)
        P = (T - phi)*self.beta
        P = th.sum(th.exp(-P*(T - phi)),0)

        return P

    def getUncertainty(self, X):
        '''
        Calculates the variance of the current model at a given set of points
        Args:
            X (th.DoubleTensor) = Nx1 column vector of training points X-coords
        Returns:
            sigma (th.DoubleTensor) = Nx1 column vector of the variance at each provided X coord.
        '''
        phi = self.getBasis(X)
        sigma = th.DoubleTensor(X.size(0)).zero_()

        for idx, val in enumerate(X):
            phi0 = phi[idx,:].unsqueeze(1)
            sigma0 = th.mm(th.transpose(phi0,0,1), th.mm(self.s0, phi0)).squeeze()
            #Eq. 3.59
            sigma[idx] = 1.0/self.beta + sigma0[0]
        return sigma

    def getLogEvidence(self, X, T):
        '''
        Calculates the log evidence of the given bayesian model
        Args:
            X (th.DoubleTensor) = Nx1 column vector of test points X-coords
            T (th.DoubleTensor) = Nx1 matrix of test values with L different data sets
        Returns:
            logErr (Variable) = log of the marginal likelihood function or evidence
        '''
        phi = self.getBasis(X)
        N = T.size(0)
        #Eq. 3.81
        A = self.alpha*th.eye(self.m).type(th.DoubleTensor) + self.beta*th.mm(th.transpose(phi, 0, 1), phi)
        #Eq. 3.84
        mn = self.beta*th.mm(th.mm(th.inverse(A), th.transpose(phi,0,1)), T)
        #Eq. 3.82
        E_mn = (self.beta/2.0)*th.norm(T - th.mm(phi, mn))**2 + (self.alpha/2.0)*th.mm(th.transpose(mn,0,1), mn)
        #Eq. 3.86
        return (self.m/2.0)*np.log(self.alpha) + (N/2.0)*np.log(self.beta) - Variable(E_mn).data.numpy() \
        - 0.5*np.log(np.linalg.det(Variable(A).data.numpy())) - (N/2.0)*np.log(2.0*np.pi)

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
        phi = th.mm(self.getBasis(X),self.m0)
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
    #X = th.linspace(0,1,N).unsqueeze(1).type(th.DoubleTensor)
    X = th.rand(N,L).type(th.DoubleTensor) #Random X here
    mu = th.sin(2.0*np.pi*X).expand(N,L)
    if(std > 0):
        T = th.normal(mu,std).type(th.DoubleTensor)
    else:
        T = mu.type(th.DoubleTensor)

    return [X, T]

def plotRegression(ax, blr):  
    #Plot data points
    X = th.linspace(0,1,100).unsqueeze(1).type(th.DoubleTensor)
    phi = blr.getBasis(X)
    sigma = blr.getUncertainty(X)
    w = blr.getWeightsMAP()
    Y = th.mm(phi,w)
        
    x_np = Variable(X.squeeze()).data.numpy()
    y_np = Variable(Y.squeeze()).data.numpy()
    std_np = Variable(th.pow(sigma,0.5).squeeze()).data.numpy()

    ax.plot(x_np, y_np, '-r')
    ax.fill_between(x_np, y_np+std_np, y_np-std_np, facecolor='r',alpha=0.3)

    #Plot true function
    y_np = np.sin(2.0*np.pi*x_np)
    ax.plot(x_np, y_np, '-g')


def plotScatter(ax, blr, X_train, T_train, idx):  
    #Plot data points
    if(idx >= 0):
        x = Variable(X_train[0:idx+1]).data.numpy()
        y = Variable(T_train[0:idx+1]).data.numpy()
        ax.plot(x, y, 'o', markersize=7, \
        markeredgewidth=1,markeredgecolor='b', \
          markerfacecolor='None')

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    f, ax = plt.subplots(2, 2, figsize=(8, 8))
    f.suptitle('Figure 3.8, pg. 157', fontsize=14)
    
    m = 10 #Number of basis
    alpha = 2.0
    mu = th.DoubleTensor(m,1).zero_()
    beta = 25
    blr = BayesianLinearReg(m, mu, alpha, beta, basis='guassian')
    X_train, T_train = generateData(1, 25, 0.3)
    
    #Single Training point
    x_n = th.DoubleTensor([X_train[0,0]]).unsqueeze(1)
    t_n = th.DoubleTensor([T_train[0,0]]).unsqueeze(1)
    blr.posterUpdate(x_n, t_n)

    plotScatter(ax[0,0], blr, X_train, T_train, 0)
    plotRegression(ax[0,0], blr)

    #Two Training point
    x_n = X_train[0:2,0].unsqueeze(1)
    t_n = T_train[0:2,0].unsqueeze(1)
    blr.posterUpdate(x_n, t_n)

    plotScatter(ax[0,1], blr, X_train, T_train, 1)
    plotRegression(ax[0,1], blr)

    #Four training points
    x_n = X_train[0:4,0].unsqueeze(1)
    t_n = T_train[0:4,0].unsqueeze(1)
    blr.posterUpdate(x_n, t_n)

    plotScatter(ax[1,0], blr, X_train, T_train, 3)
    plotRegression(ax[1,0], blr)

    #All 25 training points
    x_n = X_train[:,0].unsqueeze(1)
    t_n = T_train[:,0].unsqueeze(1)
    blr.posterUpdate(x_n, t_n)

    plotScatter(ax[1,1], blr, X_train, T_train, 24)
    plotRegression(ax[1,1], blr)

    for (m,n), subplot in np.ndenumerate(ax):
        ax[m,n].set_xlim([0,1])
        ax[m,n].set_ylim([-1.5,1.5])
        ax[m,n].set_xlabel(r'$x$')
        ax[m,n].set_ylabel(r'$t$',rotation=0)
    plt.tight_layout(rect=[0,0, 1, 0.93])
    #plt.savefig('Figure3_8.png')
    plt.show()