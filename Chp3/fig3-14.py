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

def polyModel(X, m):
        exp = th.linspace(0, m-1, m).unsqueeze(0).expand(X.size(0), m)
        phi = th.pow(X.expand(X.size(0),m),exp)
        return phi

def guassianModel(X, mu, m):
    #X = number of points
    s = 1.0/m
    mu = th.transpose(mu,0,1)
    phi = th.FloatTensor(X.size(0),m).zero_() + 1

    phi0 = th.pow(X.expand(X.size(0),m-1)-mu.expand(X.size(0),m-1),2)
    phi[:,1::] = th.exp(-phi0/(2.0*s**2))
    return phi

def generateData(L,N,std):
    X1 = th.linspace(0,1,N).unsqueeze(1)
    mu = th.sin(2.0*np.pi*X1)
    T = th.normal(mu,std)

    return [X1, T]

def getLogEvidence(X, T, m, alpha, beta):
    phi = polyModel(X, m)
    N = T.size(0)
    A = alpha*th.eye(m) + beta*th.mm(th.transpose(phi, 0, 1), phi)
    mn = beta*th.mm(th.mm(th.inverse(A), th.transpose(phi,0,1)), T)
    E_mn = (beta/2.0)*th.norm(T - th.mm(phi, mn))**2 + (alpha/2.0)*th.mm(th.transpose(mn,0,1), mn)

    return (m/2.0)*np.log(alpha) + (N/2.0)*np.log(beta) - Variable(E_mn).data.numpy() \
    - 0.5*np.log(np.linalg.det(Variable(A).data.numpy())) - (N/2.0)*np.log(2.0*np.pi)

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    f = plt.figure(figsize=(8, 4))
    f.suptitle('Figure 3.14, pg. 168', fontsize=14)
    
    m = 10 #Number of basis
    mu = th.FloatTensor(m,1).zero_()
    alpha = 0.005
    beta = 9
    X_train, T_train = generateData(1, 10, 0.1)

    X = range(0,10)
    Y = np.zeros(10)

    for i in range(0,10):
        Y[i] = getLogEvidence(X_train, T_train, i+1, alpha, beta)

    plt.plot(X, Y, '-b')
    plt.xlim([-1, 10])
    plt.xlabel(r'$W$')
    plt.ylabel(r'$ln ~p\left(\mathbf{t} |\alpha,\beta \right )$')
    #plt.savefig('Figure3_14.png')
    plt.show()