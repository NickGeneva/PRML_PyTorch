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
    def __init__(self, m, lmbda):
        self.m = m
        self.lmbda = lmbda
        
        if(m > 2):
            self.mu = th.linspace(0,1,self.m-1).unsqueeze(1)
            self.w0 = th.linspace(0,1,self.m).unsqueeze(1)
        elif(m > 1):
            self.mu = th.FloatTensor([[0.5]])
            self.w0 = th.linspace(0,1,self.m).unsqueeze(1)
        else:
            self.mu = 0
            self.w0 = th.FloatTensor([[0]])

    def calcRegression(self, X, T):
        phi = self.polyModel(X).type(th.DoubleTensor)

        w = th.mm(th.transpose(phi,0,1),T.type(th.DoubleTensor)) #NxM matrix (multi-output approach)
        w2 = th.mm(th.transpose(phi,0,1),phi) #MxM matrix
        w2 = th.inverse(self.lmbda*th.eye(self.m).type(th.DoubleTensor) + w2)
        self.w0 = th.mm(w2,w)

    def polyModel(self, X):
        s = 1.0/self.m
        if(self.m > 1):
            exp = th.linspace(0,self.m-1,self.m).unsqueeze(0).expand(X.size(0),self.m)
            phi = th.pow(X.expand(X.size(0),self.m), exp)
        else:
            exp = th.FloatTensor([[0]])
            phi = th.pow(X.expand(X.size(0),self.m),0)
        return phi

    def guassianModel(self, X):
        #X = number of points
        s = 1.0/(self.m-1)
        mu = th.transpose(self.mu,0,1)
        phi = th.FloatTensor(X.size(0),self.m).zero_() + 1
  
        phi0 = th.pow(X.expand(X.size(0),self.m-1)-mu.expand(X.size(0),self.m-1),2)
        phi[:,1::] = th.exp(-phi0/(2.0*s**2))
        return phi

    def getWeights(self):
        return self.w0

    def getTestError(self, X, T):
        N = T.size(0)
        phi = th.mv(self.guassianModel(X),self.w0)
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
    f, ax = plt.subplots(2, 2, figsize=(8, 7))
    f.suptitle('Figure 1.4, pg. 7', fontsize=14)
    
    lmbda = np.exp(-50)
    X_train, T_train = generateData(1, 10, 0.3)
    X_test, T_test = generateData(1, 100, 0)

    X0 = np.linspace(0, 1, 100)
    M = np.array([[1,2],[4,10]])
    
    for (m,n), ax0 in np.ndenumerate(ax):
        lsr = LeastSquaresReg(M[m, n],lmbda)
        lsr.calcRegression(X_train,T_train)
        
        X_tensor = th.FloatTensor(X0).unsqueeze(1)
        Y0 = Variable(th.mm(lsr.polyModel(X_tensor).type(th.DoubleTensor), lsr.getWeights())).data.numpy()

        ax0.plot(Variable(X_test).data.numpy(), Variable(T_test).data.numpy(), '-g')
        ax0.plot(Variable(X_train).data.numpy(), Variable(T_train).data.numpy(), 'o', markersize=7, \
        markeredgewidth=1,markeredgecolor='b', \
          markerfacecolor='None')
        ax0.plot(X0, Y0, '-r')

        ax0.text(0.6, 0.3, r'$M='+str(M[m, n]-1)+'$')
        ax0.set_xlim([-0.1, 1.1])
        ax0.set_ylim([-1.5, 1.5])
        ax0.set_xlabel(r'$x$')
        ax0.set_ylabel(r'$t$')
    
    plt.tight_layout(rect=[0,0, 1.0, 0.93])
    #plt.savefig('Figure1_4.png')
    plt.show()
