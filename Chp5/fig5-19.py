'''
Nicholas Geneva
ngeneva@nd.edu
August 15, 2017
'''
import sys

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib import rc
from torch.autograd import Variable

dtype = th.DoubleTensor

class simpleNN():
    def __init__(self, D_in, H, D_out, learning_rate):
        """Class for a simple 2 layer neural network
        Args:
            D_in (Int) = Number of input parameters
            H (Int) = Number of hidden paramters
            D_out (Int) = Number of output parameters
        """
        self.model = th.nn.Sequential(
            th.nn.Linear(D_in, H),
            th.nn.Tanh(),
            th.nn.Linear(H, D_out),
        ).double()

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
            err = th.randn(1).type(dtype)

            #Feed inputes into neural network
            y_pred = self.model(x_t)

            #Now lets compute out loss
            loss = self.loss_fn(y_pred, y_t)
            err += loss.data

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
            t_pred (th.DoubleTensor): [N x D_out]
        """
        x_t = Variable(x_train, requires_grad=False)
        t_pred = self.model(x_t)
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

def generateData(L,N,std):
    """Generates a set of synthetic data evenly distributed along the X axis
        with a target function of (2(x - 0.5))^3 + 0.75*(x-0.5)^2 - x + 1.0 with 
        guassian noise in the Y direction
    Args:
        L (Int) = Number of data sets desired
        N (Int) = Number of points in data set
        std (Array) = standard deviation of guassian noise
    Returns:
        X (th.DoubleTensor) = [N x L] matrix of X coords of target points
        T (th.DoubleTensor) = [N x L] matrix of Y coord of target points
    """
    X = th.linspace(0,1,N).unsqueeze(1).type(dtype)
    mu =  (th.pow(2*(X-0.5),3) + th.pow(X-0.5,2) - 0.75*(X-0.5) + 0.5).expand(N,L)
    if(std > 0):
        T = th.normal(mu,std).type(th.DoubleTensor)
    else:
        T = mu.type(th.DoubleTensor)

    return X,T

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    #Set up subplots
    f, ax = plt.subplots(1, 2, figsize=(9, 5))
    f.suptitle('Figure 5.19 pg. 273', fontsize=14)

    X_train,T_train = generateData(1,100,0.05)
    X_test,T_test = generateData(1,100,0)
    hidden_units = [1, 3, 10] #hidden units to use
    D_in, H, D_out = 1, 10, 1 #Dimension in, Hidden units, Dimension out 
    lr, err = 1e-3, 1e-4 #learning rate, error threshold

    sNN = simpleNN(D_in, H, D_out, lr)
    sNN.trainNN(X_train, T_train, err)
    y_pred = sNN.getTPred(X_test)
    ax[0].scatter(Variable(X_train[:,0]).data.numpy(), Variable(T_train[:,0]).data.numpy(), s=20, marker='o', facecolors='none', edgecolors='g')
    ax[0].plot(Variable(X_test).data.numpy(), y_pred.data.numpy(), '-r')

    #Transform training data such that there are multiple solutions at one given input
    lr, err = 1e-3, 1e-5 #learning rate, error threshold
    sNN = simpleNN(D_in, H, D_out, lr)
    sNN.trainNN(T_train, X_train, err)
    y_pred = sNN.getTPred(X_test)
    ax[1].scatter(Variable(T_train[:,0]).data.numpy(), Variable(X_train[:,0]).data.numpy(), s=20, marker='o', facecolors='none', edgecolors='g')
    ax[1].plot(Variable(X_test).data.numpy(), y_pred.data.numpy(), '-r')

    for n, subplot in np.ndenumerate(ax):
        ax[n].set_xlim([-0.1,1.1])
        ax[n].set_xticks([0, 1]) 
        ax[n].set_ylim([-0.1,1.1])
        ax[n].set_yticks([0, 1]) 
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5, rect=[0,0, 1, 0.9])
    #plt.savefig('Figure5_19.png')
    plt.show()