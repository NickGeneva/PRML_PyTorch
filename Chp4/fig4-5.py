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
            W (np.array) = [3, 3] array of basis weights
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
        m = np.zeros(3)
        b = np.zeros(3)
        p = np.array([[0, 1, 2], [1, 2, 0]])

        for i in range(3):
            m[i] = -(W[1,p[0,i]]-W[1,p[1,i]])/(W[2,p[0,i]]-W[2,p[1,i]])
            b[i] = -(W[0,p[0,i]]-W[0,p[1,i]])/(W[2,p[0,i]]-W[2,p[1,i]])

        return m, b, Variable(W).data.numpy()
        
    def classLogRegression(self, X0, T):
        '''
        Perform log regression classifications with explicit derivative since it can be found
        Args: (N = num of points, K = num of classes)
            X (th.DoubleTensor) = [N, 2] Tensor of X and Y coords of data
            T (th.DoubleTensor) = [N, K] Tensor of classification target vectors
        Returns:
            W (np.array) = [3, 3] array of basis weights
        '''
        N = T.size(0)
        K = T.size(1)

        X = th.DoubleTensor(N,3).zero_() #Add column of 1s for bias term
        X[:,0] = 1
        X[:,1:] = X0

        W_new = th.DoubleTensor(3,3).zero_() + 1
        W_old = th.DoubleTensor(3,3).zero_()
        y_n = th.DoubleTensor(3).zero_()


        lr = 1e-4 #Learning rate
        err = 1e-5 #Stop criteria

        #AVOID LOOPS WHEN POSSIBLE! TENSOR CALCULATIONS ARE FAST!
        #while (th.norm(W_new-W_old)/th.norm(W_new) > err):
        #    W_old = W_new
        #    d_Err = th.DoubleTensor(3,3).zero_()
        #    for n in range(N):
        #        phi_n = X[n,:]
        #        a_i = W_old.mv(phi_n)
        #        y_n = th.exp(a_i)/(th.exp(a_i).sum())
        #        d_Err = d_Err + (y_n - T[n,:]).unsqueeze(1).mm(phi_n.unsqueeze(1).t())
        #    W_new = W_old - lr*d_Err
    
        while (th.norm(W_new-W_old)/th.norm(W_new) > err):
            W_old = W_new
            a_i = W_old.mm(X.t())
            y_n = th.exp(a_i)/(th.exp(a_i).sum(0).expand(a_i.size())) #Eq. 4.62
            W_new = W_old - lr*(y_n - T.t()).mm(X) #Eq. 4.91 and 3.22

        return Variable(W_new).data.numpy()

    def classLogRegressionAutograd(self, X0, T):
        '''
        Perform log regression classifications using PyTorch's autograd
        Much slower than the explicit derivative version
        Args: (N = num of points, K = num of classes)
            X (th.DoubleTensor) = [N, 2] Tensor of X and Y coords of data
            T (th.DoubleTensor) = [N, K] Tensor of classification target vectors
        Returns:
            W (np.array) = [3, 3] array of basis weights
        '''
        N = T.size(0)
        K = T.size(1)

        X = th.DoubleTensor(N,3).zero_() #Add column of 1s for bias term
        X[:,0] = 1
        X[:,1:] = X0

        #Wrap/create all used tensors in Variables so we can track operations and compute
        X = Variable(X, requires_grad=False)
        T = Variable(T, requires_grad=False)
        W_new = Variable(th.DoubleTensor(3,3).zero_() + 1, requires_grad=False) 
        W_old = Variable(th.DoubleTensor(3,3).zero_(), requires_grad=True)

        lr = 1e-4 #Learning rate
        err = 1e-5 #Stop criteria

        while (th.norm(W_new.data - W_old.data)/th.norm(W_new.data) > err):
            W_old.data = W_new.data
            a_i = W_old.mm(X.t())
            y_n = th.exp(a_i)/(th.exp(a_i).sum(0).expand(a_i.size())) #Eq. 4.62
            #Compute loss function (least squares)
            loss = th.pow(y_n - T.t(), 2).sum()
            #Zero out gradient data so we can backward pass
            if W_old.grad is not None:
                W_old.grad.data.zero_()
            #Backward pass to compute gradient of all variables involved
            loss.backward()
            #Now update the weight with our computed
            W_new.data = W_old.data - lr*W_old.grad.data #Eq. 4.91 and 3.22

        return W_new.data.numpy()

def generate3ClassData(N, outlier=8):
    '''
    Geneate data consisting of 3 classes to classify
    Args:
        N (Int) = Number of points in each class
    Returns:
        X (th.DoubleTensor) = [3*N, 3] X and Y coords of data
        T (th.DoubleTensor) = [3*N, 3] Classification target vector 
    '''
    X_data = th.DoubleTensor(3*N, 2)
    T_data = th.DoubleTensor(3*N, 3).zero_()

    #X coordinates
    X_data[:N,0] = -2 + 1.5*th.randn(N).double()
    X_data[N:2*N,0] = 1.0*th.randn(N).double()
    X_data[2*N:,0] = 2 + 1.25*th.randn(N).double()
    #Y coordinates
    X_data[:N,1] = X_data[:N,0] + 1.0*th.abs(th.randn(N).double()) + 3.0
    X_data[N:2*N,1] = X_data[N:2*N,0] + 1.0*th.abs(th.randn(N).double())
    X_data[2*N:,1] = X_data[2*N:,0] - 1.0*th.abs(th.randn(N).double()) - 2.0

    #Target vector creation
    T_data[:N, 0] = 1
    T_data[N:2*N, 1] = 1
    T_data[2*N:, 2] = 1

    return X_data, T_data

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    f, ax = plt.subplots(1, 2, figsize=(8, 4))
    f.suptitle('Figure 4.5, pg. 187', fontsize=14)
    
    N = 40
    K = 3
    outliers = 8
    X, T = generate3ClassData(N, outliers)
    cl = classify()
    
    #Seperate out classes for plotting
    X0 = np.zeros((K, N))
    Y0 = np.zeros((K, N))
    c1 = 0
    c2 = 0
    for i in range(X.size(0)):
        if(T[i,0] == 1): #Class 1
            X0[0, c1] = X[i,0]
            Y0[0, c1] = X[i,1]
            c1 = c1 + 1
        elif(T[i,1] == 1): #Class 2
            X0[1, c2] = X[i,0]
            Y0[1, c2] = X[i,1]
            c2 = c2 + 1
        else: #Class 3
            X0[2, i-(c1+c2)] = X[i,0]
            Y0[2, i-(c1+c2)] = X[i,1]

    #Least squares
    m, b, w = cl.classLeastSquares(X[:,:],T[:,:])
    X_c, Y_c = np.meshgrid(np.linspace(-6,6,150), np.linspace(-6,6,150))
    Z = np.zeros(X_c.shape)
    for (x,y), val in np.ndenumerate(X_c):
        phi1 = w[0,0] + w[1,0]*X_c[x,y] + w[2,0]*Y_c[x,y]
        phi2 = w[0,1] + w[1,1]*X_c[x,y] + w[2,1]*Y_c[x,y]
        phi3 = w[0,2] + w[1,2]*X_c[x,y] + w[2,2]*Y_c[x,y]
        mx = np.argmax([phi1, phi2, phi3])
        if(mx == 0): #Class 1
            Z[x,y] = 0
        elif(mx == 1): #Class 2
            Z[x,y] = 10
        else: #Class 3
            Z[x,y] = 20

    #Color classification regions, for some reason need an extra level between class 2 and 3
    c = ('r','g','b')
    ax[0].contourf(X_c, Y_c, Z, levels = [-5,5,15,20], colors=c, alpha=0.3)
    #ax[0].contour(X_c, Y_c, Z, levels = [-5,5,15,20], colors='k', linewidths=0.8)
    ax[0].scatter(X0[0,:], Y0[0,:], c='r', marker='x')
    ax[0].scatter(X0[1,:], Y0[1,:], c='g', marker='+')
    ax[0].scatter(X0[2,:], Y0[2,:], marker='o', facecolors='none', edgecolors='b')
    ax[0].set_title('Least Squares')

    #Log Regression
    w = cl.classLogRegression(X,T)
    X_c, Y_c = np.meshgrid(np.linspace(-6,6,150), np.linspace(-6,6,150))
    Z = np.zeros(X_c.shape)
    for (x,y), val in np.ndenumerate(X_c):
        a = w.dot([1, X_c[x,y], Y_c[x,y]])

        phi1 = np.exp(a[0])/(np.exp(a).sum())
        phi2 = np.exp(a[1])/(np.exp(a).sum())
        phi3 = np.exp(a[2])/(np.exp(a).sum())
        mx = np.argmax([phi1, phi2, phi3])
        if(mx == 0): #Class 1
            Z[x,y] = 0
        elif(mx == 1): #Class 2
            Z[x,y] = 10
        else: #Class 3
            Z[x,y] = 20

    #Color classification regions, for some reason need an extra level between class 2 and 3
    c = ('r','g','b')
    ax[1].contourf(X_c, Y_c, Z, levels = [-5,5,15,20], colors=c, alpha=0.3)
    #ax[1].contour(X_c, Y_c, Z, levels = [-5,5,15,20], colors='k', linewidths=0.8)
    ax[1].scatter(X0[0,:], Y0[0,:], c='r', marker='x')
    ax[1].scatter(X0[1,:], Y0[1,:], c='g', marker='+')
    ax[1].scatter(X0[2,:], Y0[2,:], marker='o', facecolors='none', edgecolors='b')
    ax[1].set_title('Logistic Regression')

    for n, ax0 in enumerate(ax):
        ax0.set_xlim([-6,6])
        ax0.set_ylim([-6,6])
    plt.tight_layout(rect=[0,0, 1.0, 0.93])
    #plt.savefig('Figure4_5.png')
    plt.show()
