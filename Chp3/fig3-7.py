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

class BayesianLinearReg():
    def __init__(self, m, m0, s0, beta):

        self.m = m
        self.m0 = m0
        self.s0 = s0
        self.beta = beta
    
    def posterUpdate(self, X, T):
        
        phi = self.polyModel(X)
        sn_i = th.inverse(self.s0) + self.beta*th.mm(th.transpose(phi,0,1),phi)
        sn = th.inverse(sn_i)
        mn = th.mm(th.inverse(self.s0),self.m0) + self.beta*th.mv(th.transpose(phi,0,1),T.squeeze())
        mn = th.mm(sn,mn)

        #Our updated mean and covariance is now our prior
        self.m0 = mn
        self.s0 = sn

    def update_posterior(self, x_, t_):
        '''
        Updates self.mn, self.sn given training points x_, t_
        '''
        # eqn (3.50) (3.51)
        s0 = Variable(self.s0).data.numpy()
        m0 = Variable(self.m0).data.numpy()

        self.sn = np.linalg.inv(np.linalg.inv(s0) + self.beta * x_.T.dot(x_))
        mn0 = np.linalg.inv(s0).dot(m0) + self.beta * x_.T.dot(t_)
        print(mn0)
        self.mn = self.sn.dot(np.linalg.inv(s0).dot(m0) + self.beta * x_.T.dot(t_))

        #self.posterior = mv_norm(mean=self.mn, cov=self.sn)
        
        self.m0 = th.FloatTensor(self.mn)
        self.s0 = th.FloatTensor(self.sn)
        

    def polyModel(self, X):
        exp = th.linspace(0,self.m-1,self.m).unsqueeze(0).expand(X.size(0),self.m)
        phi = th.pow(X.expand(X.size(0),self.m),exp)
        return phi

    def getWeights(self):
        mu = Variable(self.m0.squeeze()).data.numpy()
        covar =  Variable(self.s0.squeeze()).data.numpy()
        norm = np.random.multivariate_normal(mu, covar)
        return th.FloatTensor(norm).unsqueeze(1)

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

def generateData(L,N,a0,a1,std):
    """Generates a set of synthetic data with X values taken from a uniform distribution
        and a linear target function and guassian noise in the Y direction

    Args:
        L (Variable) = Number of data sets desired
        N (Variable) = Number of points in data set
        std (Array) = standard deviation of guassian noise

    Returns:
        X (th.FloatTensor) = NxL matrix of X coords of target points
        T (th.FloatTensor) = NxL matrix of Y coord of target points
    """
    X1 = 2.0*th.rand(N,L) - 1.0
    mu = (a0 + a1*X1)
    T = th.normal(mu,std)

    return [X1, T]

def plotPrior(ax, blr):
    x0 = np.linspace(-1, 1, 100)
    y0 = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x0, y0)
    P = np.zeros((len(x0),len(y0)))

    for idx, valx in enumerate(x0):
        for idy, valy  in enumerate(y0):
            w = th.FloatTensor([[valx],[valy]])
            p = blr.getPrior(w)
            P[idy,idx] = Variable(p).data.numpy()

    ax.plot([-0.3],[0.5], '+', markersize=10, \
        markeredgewidth=2, markeredgecolor='w', \
          markerfacecolor='w')
    #Contour Plot of Guassian
    CS = ax.contourf(X, Y, P, 20, cmap=plt.cm.jet)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_xlabel(r'$w_0$')
    ax.set_ylabel(r'$w_1$')

def plotLikelyhood(ax, blr, x_n, t_n):
    
    x0 = np.linspace(-1, 1, 100)
    y0 = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x0, y0)
    P = np.zeros((len(x0),len(y0)))

    for idx, valx in enumerate(x0):
        for idy, valy  in enumerate(y0):
            w = th.FloatTensor([[valx],[valy]])
            p = blr.getLikelihood(w, x_n, t_n)
            P[idy,idx] = Variable(p).data.numpy()

    #Contour Plot of Guassian
    CS = ax.contourf(X, Y, P, 20, cmap=plt.cm.jet)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_xlabel(r'$w_0$')
    ax.set_ylabel(r'$w_1$')

def plotScatter(ax, blr, X_train, T_train, idx):
    
    #Plot read lines
    x0 = np.linspace(-1, 1, 100)
    for i in range(0,6):
        w = blr.getWeights().squeeze()
        y0 = w[0] + w[1]*x0
        ax.plot(x0,y0,'-r')
    
    #Plot data points
    if(idx >= 0):
        x = Variable(X_train[0:idx+1]).data.numpy()
        y = Variable(T_train[0:idx+1]).data.numpy()
        ax.plot(x, y, 'o', markersize=7, \
        markeredgewidth=1,markeredgecolor='b', \
          markerfacecolor='None')

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    f, ax = plt.subplots(4, 3, figsize=(8, 10))
    f.suptitle('Figure 3.7, pg. 155', fontsize=14)
    
    dim = 2
    alpha = 2.0
    mu = th.FloatTensor(dim,1).zero_()
    s0 = (1/alpha)*th.eye(dim)
    beta = 25
    blr = BayesianLinearReg(2, mu, s0, beta)
    X_train, T_train = generateData(1, 20, -0.3, 0.5, 0.2)

    for idx, val in enumerate(T_train):
        if(idx == 0):
            plotPrior(ax[0,1], blr)
            plotScatter(ax[0,2], blr, X_train, T_train, idx-1)
            x_n = th.FloatTensor([X_train[idx,0]])
            t_n = th.FloatTensor([T_train[idx,0]])
            blr.posterUpdate(x_n, t_n)
        
        if(idx > 0 and idx < 3):
            x_n0 = th.FloatTensor([X_train[idx-1,0]])
            t_n0 = th.FloatTensor([T_train[idx-1,0]])
            plotLikelyhood(ax[idx,0], blr, x_n0, t_n0)
            plotPrior(ax[idx,1], blr)
            plotScatter(ax[idx,2], blr, X_train, T_train, idx-1)

        if(idx == len(T_train)-1):
            x_n0 = th.FloatTensor([X_train[idx-1,0]])
            t_n0 = th.FloatTensor([T_train[idx-1,0]])
            plotLikelyhood(ax[3,0], blr, x_n0, t_n0)
            plotPrior(ax[3,1], blr)
            plotScatter(ax[3,2], blr, X_train, T_train, idx-1)
        
        x_n = th.FloatTensor([X_train[idx,0]])
        t_n = th.FloatTensor([T_train[idx,0]])
        blr.posterUpdate(x_n, t_n)
        
    ax[0,0].axis('off')
    ax[0,0].set_title('likeihood')
    ax[0,1].set_title('prior/posterior')
    ax[0,2].set_title('data space')
    plt.tight_layout(rect=[0,0, 1, 0.93])
    #plt.savefig('Figure3_7.png')
    plt.show()