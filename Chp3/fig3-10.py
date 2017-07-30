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
    def __init__(self, m, m0, s0, beta):

        self.m = m
        self.m0 = m0
        self.s0 = s0
        self.beta = beta
        self.mu = th.linspace(-1,1,self.m-1).unsqueeze(1)
    
    def posterUpdate(self, X, T):
        phi = self.guassianModel(X)

        sn_i = th.inverse(self.s0) + self.beta*th.mm(th.transpose(phi,0,1),phi)
        sn = th.inverse(sn_i)
        mn = th.mv(th.inverse(self.s0),self.m0.squeeze()) + self.beta*th.mv(th.transpose(phi,0,1),T.squeeze())
        mn = th.mv(sn,mn)

        #Our updated mean and covariance is now our prior
        self.m0 = mn
        self.s0 = sn

    def polyModel(self, X):
        exp = th.linspace(0,self.m-1,self.m).unsqueeze(0).expand(X.size(0),self.m)
        phi = th.pow(X.expand(X.size(0),self.m),exp)
        return phi

    def guassianModel(self, X):
        #X = number of points
        s = 1.0/self.m
        mu = th.transpose(self.mu,0,1)
        phi = th.FloatTensor(X.size(0),self.m).zero_() + 1
  
        phi0 = th.pow(X.expand(X.size(0),self.m-1)-mu.expand(X.size(0),self.m-1),2)
        phi[:,1::] = th.exp(-phi0/(2.0*s**2))
        return phi

    def getWeights(self):
        mu = Variable(self.m0.squeeze()).data.numpy()
        covar =  Variable(self.s0.squeeze()).data.numpy()
        norm = np.random.multivariate_normal(mu, covar)
        return th.FloatTensor(norm).unsqueeze(1)

    def getm0(self):
        return self.m0

    def getKernel(self, X, X_n):
        phi = self.guassianModel(X)
        phi_n = self.guassianModel(X_n)

        k = th.mm(self.s0, th.transpose(phi,0,1))
        return self.beta*th.mm(phi_n, k)

    def getPrior(self, W):
        P = th.mm(th.inverse(self.s0), (W-self.m0))
        P = th.mm(th.transpose(W-self.m0,0,1), P)
        P = th.exp(-0.5*P)
        return P

    def getLikelihood(self, w, x, t):
        #W = input weights [2x1]
        #x = new X training data [1x1]
        #t = new Y training data [1x1]
        phi = th.mm(self.polyModel(x),w)
        P = (t - phi)*self.beta
        P = th.sum(th.exp(-P*(t - phi)),0)

        return P

    def getUncertainty(self, X):
        phi = self.guassianModel(X)
        sigma = th.FloatTensor(X.size(0)).zero_()

        for idx, val in enumerate(X):
            phi0 = phi[idx,:].unsqueeze(1)
            sigma0 = th.mm(th.transpose(phi0,0,1), th.mm(self.s0, phi0)).squeeze()
            sigma[idx] = 1.0/self.beta + sigma0[0]
        return sigma

def generateData(L,N,std):
    """Generates a set of synthetic data with X values taken from a uniform distribution
        and a target function of sin(2*pi*x) and guassian noise in the Y direction

    Args:
        L (Variable) = Number of data sets desired
        N (Variable) = Number of points in data set
        std (Array) = standard deviation of guassian noise

    Returns:
        X (th.FloatTensor) = NxL matrix of X coords of target points
        T (th.FloatTensor) = NxL matrix of Y coord of target points
    """
    X1 = th.linspace(-1,1,N).unsqueeze(1)
    mu = th.sin(2.0*np.pi*X1)
    T = th.normal(mu,std)

    return [X1, T]

def plotKernal(ax, blr):
    x0 = np.linspace(-1, 1, 200)
    x0_n = np.linspace(-1, 1, 200)
    X, XN = np.meshgrid(x0, x0_n)

    x0 = th.linspace(-1, 1, 200).unsqueeze(1)
    x0_n = th.linspace(-1, 1, 200).unsqueeze(1)

    z0 = blr.getKernel(x0, x0_n)
    Z = Variable(z0).data.numpy()
    ax.contourf(X, XN, Z, 20, cmap=plt.cm.jet)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.axis('off')

    return Z

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    f = plt.figure(figsize=(8, 4))
    f.suptitle('Figure 3.10, pg. 159', fontsize=14)
    
    m = 10 #Number of basis
    alpha = 2.0
    mu = th.FloatTensor(m,1).zero_()
    s0 = (1.0/alpha)*th.eye(m)
    beta = 25
    blr = BayesianLinearReg(m, mu, s0, beta)
    X_train, T_train = generateData(1, 200, 0.3)
    blr.posterUpdate(X_train, T_train)

    ax_c = plt.subplot2grid((3, 2), (0, 1), rowspan=3)
    Z = plotKernal(ax_c, blr)

    #Plot contour profiles (really poor programming but it works)
    cs = ['m', 'g', 'r']
    for idx, x_n in enumerate([180, 100, 20]):
        y = (x_n-100)/100.0
        ax_c.plot([-1,1],[y,y],color=cs[idx])
        ax = plt.subplot2grid((3, 2), (idx, 0))
        ax.plot(np.linspace(-1,1,200),Z[:,x_n],color=cs[idx])
        ax.set_xlim([-1,1])
        ax.set_ylim([-0.02,0.07])

    plt.tight_layout(rect=[0,0, 1, 0.93])
    #plt.savefig('Figure3_10.png')
    plt.show()