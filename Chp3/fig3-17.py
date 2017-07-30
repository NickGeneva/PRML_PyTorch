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
    def __init__(self, m, m0, alpha, beta):

        self.m = m #Number of basis
        self.m0 = m0 #Prior means
        self.s0 = (1/alpha)*th.eye(m) #Prior covar
        self.alpha = alpha
        self.beta = beta
        self.mu = th.linspace(0,1,self.m-1).unsqueeze(1) #Array for guassian basis mean values
    
    def posterUpdate(self, X, T):
        phi = self.guassianModel(X)

        #Eq. 3.51
        sn_i = th.inverse(self.s0) + self.beta*th.mm(th.transpose(phi,0,1),phi)
        sn = th.inverse(sn_i)
        #Eq. 3.50
        mn = th.mv(th.inverse(self.s0),self.m0.squeeze()) + self.beta*th.mv(th.transpose(phi,0,1),T.squeeze())
        mn = th.mv(sn,mn)

        #Our updated mean and covariance is now our prior
        self.m0 = mn
        self.s0 = sn

    def polyModel(self, X):
        #Polynomial basis
        exp = th.linspace(0,self.m-1,self.m).unsqueeze(0).expand(X.size(0),self.m)
        phi = th.pow(X.expand(X.size(0),self.m),exp)
        return phi

    def guassianModel(self, X):
        #Guassian basis
        s = 1.0/(self.m-1)
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

    def getLogEvidence(self, X, T):
        phi = self.guassianModel(X)
        N = T.size(0)
        A = self.alpha*th.eye(self.m) + self.beta*th.mm(th.transpose(phi, 0, 1), phi)
        mn = self.beta*th.mm(th.mm(th.inverse(A), th.transpose(phi,0,1)), T)
        E_mn = (self.beta/2.0)*th.norm(T - th.mm(phi, mn))**2 + (self.alpha/2.0)*th.mm(th.transpose(mn,0,1), mn)

        return (self.m/2.0)*np.log(self.alpha) + (N/2.0)*np.log(self.beta) - Variable(E_mn).data.numpy() \
        - 0.5*np.log(np.linalg.det(Variable(A).data.numpy())) - (N/2.0)*np.log(2.0*np.pi)

    def getTestError(self, X, T):
        N = T.size(0)
        phi = th.mv(self.guassianModel(X),self.m0)
        err = th.sum(th.pow(T - phi,2), 0)/N
        return Variable(err).data.numpy()
        
def generateData(L,N,std):
    X1 = th.linspace(0,1,N).unsqueeze(1)
    mu = th.sin(2.0*np.pi*X1)
    if(std > 0):
        T = th.normal(mu,std)
    else:
        T = mu

    return [X1, T]

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    f.suptitle('Figure 3.17, pg. 172', fontsize=14)
    
    m = 10 #Number of basis
    alpha = 2.0
    beta = 11.1
    m0 = th.FloatTensor(m,1).zero_()
    X_train, T_train = generateData(1, 30, np.sqrt(1/beta))
    
    blr = BayesianLinearReg(m, m0, np.exp(-5), beta)
    phi = blr.guassianModel(X_train)
    e, v = th.eig(beta*th.mm(th.transpose(phi,0,1), phi))

    X0 = np.linspace(-10, 10, 100)
    gamma = np.zeros(len(X0))
    W0 = np.zeros((m, len(X0)))

    for idx, val in enumerate(X0):
        blr = BayesianLinearReg(m, m0, np.exp(val), beta)
        blr.posterUpdate(X_train, T_train)
        gamma[idx] = Variable(th.sum(th.div(e, np.exp(val) + e), 0)).data.numpy()[0,0]
        W0[:,idx] =  Variable(blr.getm0()).data.numpy()
    
    cmap = plt.cm.get_cmap("gnuplot")
    for i in range(m):
        ax.plot(gamma, W0[i,:], C=cmap(float(i)/m), label=r'$w_'+str(i)+'$')

    ax.set_ylim([-2.25, 2.25])
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$w_i$')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0, 0.9, 0.93])
    #plt.savefig('Figure3_17.png')
    plt.show()
