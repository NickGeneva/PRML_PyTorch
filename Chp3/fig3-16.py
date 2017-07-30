'''
Nicholas Geneva
ngeneva@nd.edu
July 21, 2017
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
        self.mu = th.linspace(0,1,self.m-1).unsqueeze(1) #means for guassian basis
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
    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 4))
    f.suptitle('Figure 3.16, pg. 172', fontsize=14)
    
    m = 10 #Number of basis
    alpha = 2.0
    beta = 11.1
    m0 = th.DoubleTensor(m,1).zero_()
    X_train, T_train = generateData(1, 30, np.sqrt(1/beta)) #Generate training data
    X_true, T_true = generateData(1, 100, 0) #Generate test set data
    
    blr = BayesianLinearReg(m, m0, np.exp(-5), beta, basis='guassian')
    phi = blr.getBasis(X_train)
    e, v = th.eig(beta*th.mm(th.transpose(phi,0,1), phi))

    X0 = np.linspace(-5, 5, 100)
    Y0 = np.zeros(len(X0))
    gamma, E_w, evid, t_err = (np.zeros(len(X0)) for i in range(4))

    for idx, val in enumerate(X0):
        blr = BayesianLinearReg(m, m0, np.exp(val), beta, basis='guassian')
        blr.posterUpdate(X_train, T_train)
        gamma[idx] = Variable(th.sum(th.div(e, np.exp(val) + e), 0)).data.numpy()[0,0]
        m_n = blr.getWeightsMAP()
        E_w[idx] = np.exp(val)*th.dot(m_n, m_n)
        evid[idx] = blr.getLogEvidence(X_train, T_train)
        t_err[idx] = blr.getTestError(X_true, T_true)
    
    ax1.plot(X0, gamma, '-r',label=r'$\gamma$')
    ax1.plot(X0, E_w, '-b', label=r'$2\alpha E_w \left(\mathbf{m}_N \right)$')
    ax1.set_ylim([0, 25])
    ax1.set_xlabel(r'$ln \, \alpha$')
    ax1.get_yaxis().set_ticklabels([])
    ax1.legend(loc=2)

    ax3 = ax2.twinx()
    line1, = ax2.plot(X0, evid, '-r', label=r'$ln~p\left(\mathbf{t}|\alpha,\beta\right)$')
    line2, = ax3.plot(X0, t_err, '-b', label=r'$E_{test}$')
    ax2.set_xlabel(r'$ln \, \alpha$')
    ax2.get_yaxis().set_ticklabels([])
    ax3.get_yaxis().set_ticklabels([])
    ax3.legend(handles=[line1, line2], loc=2)
    
    plt.tight_layout(rect=[0,0, 1, 0.93])
    #plt.savefig('Figure3_16.png')
    plt.show()
