'''
Nicholas Geneva
ngeneva@nd.edu
August 11, 2017
'''
import sys

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib import rc
from numpy.linalg import inv
from torch.autograd import Variable

dtype = th.DoubleTensor

class BabyNet(th.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        Small 2 layer Neural Net
        Args:
            D_in (Int) = Number of input parameters
            H (Int) = Number of hidden paramters
            D_out (Int) = Number of output parameters
        """
        super(BabyNet, self).__init__()
        self.linear1 = th.nn.Linear(D_in, H)
        self.f1 = th.nn.Tanh()
        self.linear2 = th.nn.Linear(H, D_out)

    def forward(self, x):
        """
        Forward pass of the neural network
        Args:
            x (th.DoubleTensor): [N x D_in] column matrix of training inputs
        Returns:
            out (th.DoubleTensor): [N x D_out] matrix of neural network outputs
        """
        lin1 = self.f1(self.linear1(x))
        out = self.linear2(lin1)
        return out

    def hiddenForward(self, x):
        """
        Calculates the value of the hidden parameters in the neural network
        Args:
            x (th.DoubleTensor): [N x D_in] column matrix of training inputs
        Returns:
            out (th.DoubleTensor): [N x H] matrix of neural network outputs
        """
        lin1 = self.f1(self.linear1(x))
        return lin1

class simpleNN():
    def __init__(self, D_in, H, D_out, learning_rate):
        """Class for a simple 2 layer neural network
        Args:
            D_in (Int) = Number of input parameters
            H (Int) = Number of hidden paramters
            D_out (Int) = Number of output parameters
        """
        self.model = BabyNet(D_in, H, D_out).double()

        for x in self.model.modules():
            if isinstance(x, th.nn.Linear):
                x.weight.data = th.normal(means=th.zeros(x.weight.size())).type(dtype)
                x.bias.data = th.zeros(x.bias.size()).type(dtype)

        self.H = H
        self.lr = learning_rate
        self.loss_fn = th.nn.MSELoss(size_average=False)
    
    def trainNN(self, x_train, y_train, err_limit):
        """
        Conduct one interation of training of the NN
        Args:
            X_train (th.DoubleTensor): [N x D_in] column matrix of training inputs
            Y_train (th.DoubleTensor): [N x D_out] column matrix of training outputs
            err_limit (float): error threshold for training the NN
        """
        x_t = Variable(x_train)
        y_t = Variable(y_train, requires_grad=False)
        idx = 0

        err0 = th.zeros(1).type(dtype)
        err = th.randn(1).type(dtype) + 1

        while (th.norm(err - err0) > err_limit):
            err0 = err
            err = th.zeros(1).type(dtype)

            #Feed inputes into neural network
            y_pred = self.model(x_t) #__call__()

            #Now lets compute out loss
            loss = self.loss_fn(y_pred, y_t)
            err = loss.data
            print(idx,err)

            # Zero the gradients before running the backward pass.
            self.model.zero_grad()
            loss.backward()

            for param in self.model.parameters():
                param.data -= self.lr * param.grad.data #Using Batch sharpest decent

            idx+=1
            if(idx > 1e6): #Give up after 1e6 attempts to train
                print('Interation break')
                break

        print('Training Loss for M='+str(self.H)+': '+str(loss.data.numpy()))

    def getTPred(self, x_train):
        """
        Get the prediction of the NN for a given set of points
        Args:
            x_train (th.DoubleTensor): [N x D_in] matrix of training inputs
        Returns:
            t_pred (Variable): [N x D_out]
        """
        x_t = Variable(x_train, requires_grad=False)
        t_pred = self.model(x_t)
        return t_pred

    def getHiddenUnits(self, x_train):
        """
        Get the hidden units 
        Args:
            x_train (th.DoubleTensor): [N x D_in] matrix of training inputs
        Returns:
            t_pred (Variable): [N x D_out]
        """
        x_t = Variable(x_train, requires_grad=False)
        t_pred = self.model.hiddenForward(x_t)
        return t_pred
        

    def getLoss(self, x_test, t_test):
        """
        Get the prediction of the NN for a given set of points
        Args:
            x_test (th.DoubleTensor): [N x D_in] matrix of test set inputs
            t_test (th.DoubleTensor): [N x D_out] matrix of test set outputs
        Returns:
            loss (Variable): loss error for the given test set
        """
        x_t = Variable(x_test, requires_grad=False)
        #Feed inputes into neural network
        t_pred = self.model(x_t)
        #Now lets compute out loss
        loss = self.loss_fn(t_pred, t_test)
        return loss

def generate2ClassData(N):
    """Generates a set of synthetic data classification data discribed in Appendix A
    Args:
        N (Int) = Number of points in each class
    Returns:
        X (th.DoubleTensor) = [2*N x 2] matrix of X and Y coords of target points
        T (th.DoubleTensor) = [2*N x 2] matrix of binary classification target vectors
    """
    X_data = th.DoubleTensor(2*N, 2)
    T_data = th.DoubleTensor(2*N, 2).zero_()

    #Generate class 1 coordinates
    mean = [-.25, 0]
    covar = [[1.0, 1.0], [0, 1.0]] #variance
    X_data[:N,:] = th.FloatTensor(np.random.multivariate_normal(mean, covar, N))

    #Generate class 2 coordinates
    mean = [1.0, -0.75]
    covar = [[1, .8], [.8, 1]]  # diagonal covariance
    mean2 = [1.0, 1]
    covar2 = [[1.0, 1.0], [0, 1.0]]  # diagonal covariance
    n = int(np.random.normal(N/2.0,5)) #Number of points from guassian 1
    X_data[N:N+n,:] = th.FloatTensor(np.random.multivariate_normal(mean, covar, n))
    X_data[N+n:,:] = th.FloatTensor(np.random.multivariate_normal(mean2, covar2, N-n))

    #Target vector creation
    T_data[:N, 0] = 1
    T_data[N:, 1] = 1

    return X_data, T_data

def plotIdealFit(ax, xlim, ylim):
    """Generates a set of synthetic data classification data discribed in Appendix A
    Args:
        N (Int) = Number of points in each class
    Returns:
        X (th.DoubleTensor) = [2*N x 2] matrix of X and Y coords of target points
        T (th.DoubleTensor) = [2*N x 2] matrix of binary classification target vectors
    """
    x = np.linspace(xlim[0],xlim[1],150)
    y = np.linspace(ylim[0],ylim[1],150)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    #Class 1
    mean0 = [-.25, 0]
    cov0 = [[1.0, 1.0], [0, 1.0]]
    #Class 2
    mean = [1.0, -0.75]
    cov = [[1, .8], [.8, 1]]
    mean2 = [0.75, 1]
    cov2 = [[1.0, 1.0], [0, 1.0]]
    
    for (i,j), val in np.ndenumerate(X):
        x = np.array([[X[i,j], Y[i,j]]])
        Z[i,j] = -5*np.exp(-0.5*(x-mean0).dot(inv(cov0)).dot((x-mean0).T)) + \
            5*np.exp(-0.5*(x-mean).dot(inv(cov)).dot((x-mean).T)) + \
            5*np.exp(-0.5*(x-mean2).dot(inv(cov2)).dot((x-mean2).T))

    #ax.contourf(X, Y, Z, 10)
    ax.contour(X, Y, Z, levels = [-5,0,5], colors='g', linewidth=0.5)

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    f.suptitle('Figure 5.4 pg. 232', fontsize=14)
    xlim, ylim = [-2.25,2.25], [-3,3]

    N = 75 #Number of points in each class
    X_train, T_train = generate2ClassData(N)
    lr, err = 1e-4, 1e-5 #learning rate, error threshold
    
    sNN = simpleNN(2, 2, 1, lr)
    sNN.trainNN(X_train, T_train[:,0], err)

    x = np.linspace(xlim[0],xlim[1],150)
    y = np.linspace(ylim[0],ylim[1],150)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((3,X.shape[0],X.shape[1]))
    for (i,j), val in np.ndenumerate(X):
        x_test = th.DoubleTensor([[X[i,j], Y[i,j]]])
        Z[0,i,j] = sNN.getTPred(x_test).data.numpy()
        Z[1:,i,j] = sNN.getHiddenUnits(x_test).data.numpy()

    sNN.getHiddenUnits(x_test)
    #Plot decision surface boundary
    ax.contour(X, Y, Z[0], levels = [-3.0,0.5,3.0], colors='r', linewidth=0.5)
    #Plot hidden activation function surface boundaries
    CS = ax.contour(X, Y, Z[1], levels = [-1.0,0.5,2.0], colors='b', linewidth=0.25)
    for c in CS.collections:
        c.set_dashes([(0, (2.0, 2.0))])  
    CS = ax.contour(X, Y, Z[2], levels = [-1.0,0.5,2.0], colors='b', linewidth=0.25)
    for c in CS.collections:
        c.set_dashes([(0, (2.0, 2.0))])

    #Seperate out classes for plotting
    X0 = np.zeros((N,2))
    Y0 = np.zeros((N,2))
    c1 = 0
    for i in range(X_train.size(0)):
        if(T_train[i,0] == 1): #Class 1
            X0[c1, 0] = X_train[i,0]
            Y0[c1, 0] = X_train[i,1]
            c1 = c1 + 1
        else: #Class 2
            X0[i-c1, 1] = X_train[i,0]
            Y0[i-c1, 1] = X_train[i,1]

    ax.scatter(X0[:,0], Y0[:,0], marker='o', facecolors='none', edgecolors='b')
    ax.scatter(X0[:,1], Y0[:,1], c='r', marker='x')
    plotIdealFit(ax, xlim, ylim)
    
    ax.set_xlim(xlim)
    ax.set_xticks([-2, -1, 0, 1, 2]) 
    ax.set_ylim(ylim)
    ax.set_yticks([-2, -1, 0, 1, 2])

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5, rect=[0,0, 1, 0.9])
    #plt.savefig('Figure5_04.png')
    plt.show()