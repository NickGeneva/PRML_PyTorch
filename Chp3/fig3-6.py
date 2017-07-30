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
        phi = self.guassianModel(X)

        w = th.mm(th.transpose(phi,0,1),T) #NxM matrix (multi-output approach)
        w2 = th.mm(th.transpose(phi,0,1),phi) #MxM matrix
        w2 = th.inverse(self.lmbda*th.eye(self.m) + w2)
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
    mu = th.sin(2.0*np.pi*X1).expand(N,L)
    if(std > 0):
        T = th.normal(mu,std)
    else:
        T = mu

    return [X1, T]

def regression(X,T,M,P,lamb):
    '''Calculates multi-output regression with guassian basis, assumes all data sets has
        the same X-coordinates.
  
      Args:
          X (th.FloatTensor) = Nx1 column vector of target points X-coords
          T (th.FloatTensor) = NxL matrix of targer values with L different data sets
          M (Variable) = Number of regression basis (includes phi0)
          P (Variable) = Number of points to calculate the solved regression function with
          lamb (Variable) = regularization term to control impact of regularization error
    Returns:
        X0 (th.FloatTensor) = X-coord of regressions
        Y (th.FloatTensor) = PxL matrix of regression points for each data set
    '''
    N = T.size(0) #Number of data points in sample
    L = T.size(1) #Number of samples

    #Set up basis functions
    mu = th.FloatTensor(1,M-1).zero_()
    mu[0,:] = th.linspace(0,1,M-1)
    s = 1.0/M #scale factor
    phi = th.FloatTensor(N,M).zero_() + 1
    phi[:,1::] = th.exp(-0.5*th.pow(X.expand(N,M-1)-mu.expand(N,M-1),2)/(s**2))
   
    #Determine regression basis weights
    w = th.mm(th.transpose(phi,0,1),T) #MxL matrix (multi-output approach)
    w2 = th.mm(th.transpose(phi,0,1),phi) #MxM matrix
    w2 = th.inverse(lamb*th.eye(M,M) + w2)
    w = th.mm(w2,w)

    #calculate regression points
    X0 = th.FloatTensor(P,1).zero_()
    X0[:,0] = th.linspace(0,1,P)
    phi = th.FloatTensor(P,M).zero_() + 1
    phi[:,1::] = th.exp(-0.5*th.pow(X0.expand(P,M-1)-mu.expand(P,M-1),2)/(s**2))
    
    return [X0, th.mm(phi,w)]

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
    Y0 = th.FloatTensor(len(gammas),P,L).zero_()
    Y1 = th.FloatTensor(len(gammas),1000,L).zero_()

    for idx, val in enumerate(gammas):
        lsr = LeastSquaresReg(M, np.exp(val))
        lsr.calcRegression(X_train,T_train)
        Y0[idx,:,:] = th.mm(lsr.guassianModel(X_train), lsr.getWeights().squeeze())
        Y1[idx,:,:] = th.mm(lsr.guassianModel(X_test), lsr.getWeights())

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
