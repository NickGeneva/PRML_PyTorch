'''
Nicholas Geneva
ngeneva@nd.edu
July 27, 2017
'''
import sys

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib import rc
from torch.autograd import Variable

class classify():
    def __init__(self):
        self.x = 0

    def classLeastSquares(self, X0, T):
        '''
        Perform least squares classifications
        Args: (N = num of points, K = num of classes)
            X (th.DoubleTensor) = [N, 2] Tensor of X and Y coords of data
            T (th.DoubleTensor) = [N, K] Tensor of classification target vectors
        Returns:
            m (Variable) = slope of decision boundary
            b (Variable) = residual of decision boundary
        '''
        N = T.size(0)
        K = T.size(1)
        X = th.DoubleTensor(N,3).zero_() #Add column of 1s for bias term
        X[:,0] = 1
        X[:,1:] = X0 
        #Eq. 4.16
        W = th.inverse(X.t().mm(X)).mm(X.t().mm(T))
        #Boundary is where activation functions are equal
        #W[2,a]Y = W[1,a]X + W[0,a]
        #Eq. 4.10
        m = -(W[1,0]-W[1,1])/(W[2,0]-W[2,1])
        b = -(W[0,0]-W[0,1])/(W[2,0]-W[2,1])

        return m, b
        
    def classLogRegression(self, X0, T):
        '''
        Perform log regression classifications with explicit derivative since it can be found
        Args: (N = num of points, K = num of classes)
            X (th.DoubleTensor) = [N, 2] Tensor of X and Y coords of data
            T (th.DoubleTensor) = [N, K] Tensor of classification target vectors
        Returns:
            m (Variable) = slope of decision boundary
            b (Variable) = residual of decision boundary
        '''
        N = T.size(0)
        K = T.size(1)

        X = th.DoubleTensor(N,3).zero_() #Add column of 1s for bias term
        X[:,0] = 1
        X[:,1:] = X0

        W_new = th.DoubleTensor(1,3).zero_() + 1
        W_old = th.DoubleTensor(1,3).zero_()

        lr = 1e-4 #Learning rate
        err = 1e-5 #Stop criteria

        #while (th.norm(W_new-W_old)/th.norm(W_new) > err):
        #    W_old = W_new
        #    d_Err = th.DoubleTensor(3).zero_()
        #    for n in range(N):
        #        phi_n = X[n,:]
        #        y_n = 1/(1 + np.exp(-W_old.dot(phi_n)))
        #        d_Err = d_Err + (y_n - T[n,0])*(phi_n)
        #    W_new = W_old - lr*d_Err

        while (th.norm(W_new-W_old)/th.norm(W_new) > err):
            W_old = W_new
            y_n = 1/(1 + th.exp(-W_old.mm(X.t()))) #Eq. 4.59
            W_new = W_old - lr*(y_n - T[:,0]).mm(X) #Eq. 4.91 and 3.22

        #Boundary is where activation functions are equal
        #W[2,a]Y = W[1,a]X + W[0,a]
        #Eq. 4.10
        m = -(W_new[0,1])/(W_new[0,2])
        b = -(W_new[0,0])/(W_new[0,2])

        return m, b

    def classLogRegressionAutograd(self, X0, T):
        '''
        Perform log regression classifications using PyTorch's autograd
        Much slower than the explicit derivative version
        Args: (N = num of points, K = num of classes)
            X (th.DoubleTensor) = [N, 2] Tensor of X and Y coords of data
            T (th.DoubleTensor) = [N, K] Tensor of classification target vectors
        Returns:
            m (Variable) = slope of decision boundary
            b (Variable) = residual of decision boundary
        '''
        N = T.size(0)
        K = T.size(1)

        X = th.DoubleTensor(N,3).zero_() #Add column of 1s for bias term
        X[:,0] = 1
        X[:,1:] = X0

        #Wrap/create all used tensors in Variables so we can track operations and compute
        X = Variable(X, requires_grad=False)
        T = Variable(T, requires_grad=False)
        W_new = Variable(th.DoubleTensor(1,3).zero_() + 1, requires_grad=False) 
        W_old = Variable(th.DoubleTensor(1,3).zero_(), requires_grad=True)

        lr = 1e-4 #Learning rate
        err = 1e-5 #Stop criteria

        while (th.norm(W_new.data - W_old.data)/th.norm(W_new.data) > err):
            W_old.data = W_new.data
            y_n = 1/(1 + th.exp(-W_old.mm(X.t()))) #Eq. 4.59
            #Compute loss function (least squares)
            loss = th.pow(y_n - T[:,0], 2).sum()
            #Zero out gradient data so we can backward pass
            if W_old.grad is not None:
                W_old.grad.data.zero_()
            #Backward pass to compute gradient of all variables involved
            loss.backward()
            #Now update the weight with our computed
            W_new.data = W_old.data - lr*W_old.grad.data #Eq. 4.91 and 3.22

        #Boundary is where activation functions are equal
        #W[2,a]Y = W[1,a]X + W[0,a]
        #Eq. 4.10
        m = -(W_new.data[0,1])/(W_new.data[0,2])
        b = -(W_new.data[0,0])/(W_new.data[0,2])

        return m, b

def generate2ClassData(N, outlier=8):
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
    X_data[:N,0] = -1+1.5*th.randn(N).double()
    X_data[N:2*N,0] = 1+0.75*th.randn(N).double()
    #Y coordinates
    X_data[:N,1] = X_data[:N,0] + 1.0*th.abs(th.randn(N).double()) + 1.25
    X_data[N:2*N,1] = X_data[N:2*N,0] - 1.25*th.abs(th.randn(N).double()) - 1.25
    #Outliers to class 2
    if(outlier > 0):
        X_data[2*N:,0] = 0.5*th.randn(outlier) + 8
        X_data[2*N:,1] = 0.75*th.abs(th.randn(outlier)) - 7 

    #Target vector creation
    T_data[:N, 0] = 1
    T_data[N:, 1] = 1

    return X_data, T_data

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    f, ax = plt.subplots(1, 2, figsize=(8, 4))
    f.suptitle('Figure 4.4, pg. 186', fontsize=14)
    
    N = 40
    K = 2
    outliers = 8
    X, T = generate2ClassData(N, outliers)
    cl = classify()
    
    #Seperate out classes for plotting
    X0 = np.zeros((K, N+outliers))
    Y0 = np.zeros((K, N+outliers))
    c1 = 0
    for i in range(X.size(0)):
        if(T[i,0] == 1): #Class 1
            X0[0, c1] = X[i,0]
            Y0[0, c1] = X[i,1]
            c1 = c1 + 1
        else: #Class 2
            X0[1, i-c1] = X[i,0]
            Y0[1, i-c1] = X[i,1]

    #No Outliers
    m, b = cl.classLeastSquares(X[:2*N,:],T[:2*N,:])
    m2, b2 = cl.classLogRegression(X[:2*N,:],T[:2*N,:])
    x_n = np.linspace(-5,9,10)

    ax[0].scatter(X0[0,:N], Y0[0,:N], c='r', marker='x')
    ax[0].scatter(X0[1,:N], Y0[1,:N], marker='o', facecolors='none', edgecolors='b')
    ax[0].plot(x_n, m*x_n + b, c='purple', label='Least Squares')
    ax[0].plot(x_n, m2*x_n + b2, c='green', label='Logistic Reg.')
    ax[0].legend(loc='lower left')

    #With Outliers
    m, b = cl.classLeastSquares(X,T)
    m2, b2 = cl.classLogRegression(X,T)
    x_n = np.linspace(-5,9,10)

    ax[1].scatter(X0[0,:N], Y0[0,:N], c='r', marker='x')
    ax[1].scatter(X0[1,:], Y0[1,:], marker='o', facecolors='none', edgecolors='b')
    ax[1].plot(x_n, m*x_n + b, c='purple', label='Least Squares')
    ax[1].plot(x_n, m2*x_n + b2, c='green', label='Logistic Reg.')


    for n, ax0 in enumerate(ax):
        ax0.set_xlim([-4,9])
        ax0.set_ylim([-9,4])
    plt.tight_layout(rect=[0,0, 1.0, 0.93])
    #plt.savefig('Figure4_4.png')
    plt.show()
