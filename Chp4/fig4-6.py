'''
Nicholas Geneva
ngeneva@nd.edu
July 28, 2017
'''
import sys

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib import rc
from numpy.linalg import inv
from torch.autograd import Variable

class classify():
    def __init__(self):
        self.x = 0

    def getMean(self, X, T):
        '''
        Gets the mean of each class
        Args: (N = num of points, K = num of classes)
            X (th.DoubleTensor) = [N, 2] Tensor of X and Y coords of data
            T (th.DoubleTensor) = [N, K] Tensor of classification target vectors
        Returns:
            m (th.DoubleTensor) = [K,2] Tensor containing X and Y of means of each class
        '''
        N = T.size(0)
        N1 = th.sum(T[:,0] == 1)
        N2 = th.sum(T[:,1] == 1)
        
        K = T.size(1)
        m = th.DoubleTensor(K,2)
        m[:,0] = (X[:,0].expand(K,N) * T.t()).sum(1)/N1
        m[:,1] = (X[:,1].expand(K,N) * T.t()).sum(1)/N2
        
        return m

    def meanProjection(self, X, T):
        '''
        Projects the data onto the line connecting the means
        Args: (N = num of points, K = num of classes)
            X (th.DoubleTensor) = [N, 2] Tensor of X and Y coords of data
            T (th.DoubleTensor) = [N, K] Tensor of classification target vectors
        Returns:
            x_m (th.DoubleTensor) = [N] Tensor containing the projection of X onto the mean basis
        '''
        N = T.size(0)
        N1 = th.sum(T[:,0] == 1)
        N2 = th.sum(T[:,1] == 1)
        M = self.getMean(X,T)
        
        W = M[1,:] - M[0,:] #Eq. 4.23
        return X.mv(W)

    def fisherMeanProjection(self, X, T):
        '''
        Projects the data onto the line connecting the means
        Args: (N = num of points, K = num of classes)
            X (th.DoubleTensor) = [N, 2] Tensor of X and Y coords of data
            T (th.DoubleTensor) = [N, K] Tensor of classification target vectors
        Returns:
            x_m (th.DoubleTensor) = [N] Tensor containing the projection of X onto the mean basis
        '''
        N = T.size(0)
        N1 = th.sum(T[:,0] == 1)
        N2 = th.sum(T[:,1] == 1)
        M = self.getMean(X,T)

        X1 = th.DoubleTensor(N1,2)
        X2 = th.DoubleTensor(N2,2)
        X1[:,0] = X[:,0].index(T[:,0].type(th.ByteTensor))
        X1[:,1] = X[:,1].index(T[:,0].type(th.ByteTensor))
        X2[:,0] = X[:,0].index(T[:,1].type(th.ByteTensor))
        X2[:,1] = X[:,1].index(T[:,1].type(th.ByteTensor))

        #Eq. 4.28
        c1 = X1-M[0,:].unsqueeze(0).expand(N1,2)
        c2 = X2-M[1,:].unsqueeze(0).expand(N2,2)
        S_w = c1.t().mm(c1) + c2.t().mm(c2)

        W = th.inverse(S_w).mv(M[1,:] - M[0,:]) #Eq. 4.23
        return X.mv(W), W

def generate2ClassData(N, outlier=0):
    '''
    Geneate data consisting of 2 classes to classify
    Args:
        N (Int) = Number of points in each class
        outlier (Int) = number of outliers to include
    Returns:
        X (th.DoubleTensor) = [2*N + outlier, 2] X and Y coords of data
        T (th.DoubleTensor) = [2*N + outlier, 2] Classification target vector 
    '''
    X_data = th.DoubleTensor(2*N + outlier, 2)
    T_data = th.DoubleTensor(2*N + outlier, 2).zero_()

    #X coordinates
    X_data[:N,0] = 1 + 1.25*th.randn(N).double()
    X_data[N:2*N,0] = 4 + 1.25*th.randn(N).double()
    #Y coordinates
    X_data[:N,1] = 0.25*X_data[:N,0] + 0.5*th.abs(th.randn(N).double()) + 2
    X_data[N:2*N,1] = 0.25*X_data[N:2*N,0] + 0.5*th.abs(th.randn(N).double())

    #Target vector creation
    T_data[:N, 0] = 1
    T_data[N:, 1] = 1

    return X_data, T_data

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    f, ax = plt.subplots(2, 2, figsize=(8, 4))
    f.suptitle('Figure 4.6, pg. 188', fontsize=14)
    
    N = 40
    K = 2
    X, T = generate2ClassData(N)
    cl = classify()
    X2 = cl.meanProjection(X, T)
    means = cl.getMean(X,T)

    #Seperate out classes for plotting
    X0 = np.zeros((K, N))
    Y0 = np.zeros((K, N))
    X_m = np.zeros((K, N))
    c1 = 0
    for i in range(X.size(0)):
        if(T[i,0] == 1): #Class 1
            X0[0, c1] = X[i,0]
            Y0[0, c1] = X[i,1]
            X_m[0, c1] = X2[i]
            c1 = c1 + 1
        else: #Class 2
            X0[1, i-c1] = X[i,0]
            Y0[1, i-c1] = X[i,1]
            X_m[1, i-c1] = X2[i]

    #Mean projection
    ax[0,0].scatter(X0[0,:N], Y0[0,:N], c='b', marker='o', s=2.0)
    ax[0,0].scatter(X0[1,:], Y0[1,:], c='r', marker='o', s=2.0)
    ax[0,0].scatter(means[0,0], means[0,1], c='b', marker='+', s=100)
    ax[0,0].scatter(means[1,0], means[1,1], c='r', marker='+', s=100)
    ax[0,0].scatter((means[0,0]+means[1,0])/2.0, (means[0,1]+means[1,1])/2.0, c='g', marker='+', s=100)
    ax[0,0].plot([means[0,0],means[1,0]],[means[0,1],means[1,1]],'-g')

    # create an inset axe in the current axe:
    ax[1,0].hist(X_m.transpose(), 20, histtype='bar', color=['b','r'], edgecolor='w', stacked=True)

    #Fisher Linear
    X2, W = cl.fisherMeanProjection(X, T)
    c1 = 0
    for i in range(X.size(0)):
        if(T[i,0] == 1): #Class 1
            X_m[0, c1] = X2[i]
            c1 = c1 + 1
        else: #Class 2
            X_m[1, i-c1] = X2[i]

    #Green line calculations
    m0 = (means[0,:]+means[1,:])/2
    A = np.array([[W[0]/W[1], 1], [-W[1]/W[0], 1]])
    b = [means[0,0]*W[0]/W[1] + means[0,1], -W[1]/W[0]*m0[0] + m0[1]]
    m4 = inv(A).dot(b)
    b = [means[1,0]*W[0]/W[1] + means[1,1], -W[1]/W[0]*m0[0] + m0[1]]
    m5 = inv(A).dot(b)

    ax[0,1].scatter(X0[0,:N], Y0[0,:N], c='b', marker='o', s=2.0)
    ax[0,1].scatter(X0[1,:], Y0[1,:], c='r', marker='o', s=2.0)
    ax[0,1].scatter(means[0,0], means[0,1], c='b', marker='+', s=100)
    ax[0,1].scatter(means[1,0], means[1,1], c='r', marker='+', s=100)
    ax[0,1].scatter((means[0,0]+means[1,0])/2.0, (means[0,1]+means[1,1])/2.0, c='g', marker='+', s=100)
    #Green lines
    xg = np.array([means[0,0], m4[0], m0[0], m5[0], means[1,0]])
    yg = np.array([means[0,1], m4[1], m0[1], m5[1], means[1,1]])
    ax[0,1].plot(xg, yg, '-g')

    # create an inset axe in the current axe:
    ax[1,1].hist(X_m.transpose(), 20, histtype='bar', color=['b','r'], edgecolor='w', stacked=True)


    for (i,j), ax0 in np.ndenumerate(ax):
        if(i == 0):
            ax0.set_xlim([-3,8])
            ax0.set_ylim([-3,4.5])
    plt.tight_layout(rect=[0,0, 1.0, 0.93])
    #plt.savefig('Figure4_4.png')
    plt.show()
