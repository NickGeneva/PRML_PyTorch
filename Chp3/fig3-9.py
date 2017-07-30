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
        self.mu = th.linspace(0,1,self.m-1).unsqueeze(1)
    
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
    X1 = th.rand(N,L)
    mu = th.sin(2.0*np.pi*X1)
    T = th.normal(mu,std)

    return [X1, T]

def plotRegression(ax, blr, plotSin = False):  
    #Plot data points
    X = th.linspace(0,1,100).unsqueeze(1)
    phi = blr.guassianModel(X)
    
    for i in range(0,5):
        w = blr.getWeights()
        Y = th.mv(phi,w.squeeze())
        x_np = Variable(X.squeeze()).data.numpy()
        y_np = Variable(Y.squeeze()).data.numpy()
        ax.plot(x_np, y_np, '-r', linewidth=1)

    #Plot true function
    if(plotSin):
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
    f.suptitle('Figure 3.9, pg. 158', fontsize=14)
    
    m = 10 #Number of basis
    alpha = 2.0
    mu = th.FloatTensor(m,1).zero_()
    s0 = (1.0/alpha)*th.eye(m)
    beta = 25
    blr = BayesianLinearReg(m, mu, s0, beta)
    X_train, T_train = generateData(1, 25, 0.3)
    
    #Single Training point
    x_n = th.FloatTensor([X_train[0,0]])
    t_n = th.FloatTensor([T_train[0,0]])
    blr.posterUpdate(x_n, t_n)

    plotRegression(ax[0,0], blr, True)
    plotScatter(ax[0,0], blr, X_train, T_train, 0)

    #Two Training point
    x_n = X_train[0:2,0].unsqueeze(1)
    t_n = T_train[0:2,0].unsqueeze(1)
    blr = BayesianLinearReg(m, mu, s0, beta)
    blr.posterUpdate(x_n, t_n)

    plotRegression(ax[0,1], blr)
    plotScatter(ax[0,1], blr, X_train, T_train, 1)

    #Four training points
    x_n = X_train[0:4,0].unsqueeze(1)
    t_n = T_train[0:4,0].unsqueeze(1)
    blr = BayesianLinearReg(m, mu, s0, beta)
    blr.posterUpdate(x_n, t_n)

    plotRegression(ax[1,0], blr)
    plotScatter(ax[1,0], blr, X_train, T_train, 3)

    #All 25 training points
    x_n = X_train[:,0].unsqueeze(1)
    t_n = T_train[:,0].unsqueeze(1)
    blr = BayesianLinearReg(m, mu, s0, beta)
    blr.posterUpdate(x_n, t_n)

    plotRegression(ax[1,1], blr)
    plotScatter(ax[1,1], blr, X_train, T_train, 24)

    for (m,n), subplot in np.ndenumerate(ax):
        ax[m,n].set_xlim([0,1])
        ax[m,n].set_ylim([-1.5,1.5])
        ax[m,n].set_xlabel(r'$x$')
        ax[m,n].set_ylabel(r'$t$',rotation=0)
    plt.tight_layout(rect=[0,0, 1, 0.93])
    #plt.savefig('Figure3_9.png')
    plt.show()