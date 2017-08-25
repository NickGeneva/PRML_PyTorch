'''
Nicholas Geneva
ngeneva@nd.edu
August 25, 2017
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

def generateTrainingData(N, std, func='para'):
    """Generates a set of training data for several abitrary functions
    Args:
        N (Int) = Number of points in each class
        std (Int) = standard deviation of the Y noise added to the data
        func (String) = training function of choice (parabolic, sine, etc.)
    Returns:
        X (th.DoubleTensor) = [N] matrix of X and Y coords of target points
        T (th.DoubleTensor) = [N] matrix of binary classification target vectors
    """
    X_data = th.linspace(-1,1,N).unsqueeze(1).type(th.DoubleTensor)
    if(func == 'para'):
        mu = X_data.pow(2)
    elif(func == 'sine'):
        mu = th.sin(np.pi*X_data)
    elif(func == 'abs'):
        mu = th.abs(X_data)
    else:
        mu = X_data > 0
    
    if(std > 0):
        T_data = th.normal(mu,std).type(th.DoubleTensor)
    else:
        T_data = mu.type(th.DoubleTensor)

    return X_data, T_data

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    #Set up subplots
    f, ax = plt.subplots(2, 2, figsize=(7, 6))
    f.suptitle('Figure 5.3 pg. 231', fontsize=14)
    xlim, ylim = [-2.25,2.25], [-3,3]

    N = 50 #Number of points in each class
    lr, err = 5e-4, 1e-6 #learning rate, error threshold
    
    #Parabolic function
    X_train, T_train = generateTrainingData(N,0,func='para')
    X_test, T_test = generateTrainingData(100,0,func='para')
    sNN = simpleNN(1, 3, 1, lr)
    sNN.trainNN(X_train, T_train, err)

    y_pred = sNN.getTPred(X_test)
    y_hidden = sNN.getHiddenUnits(X_test)
    ax[0,0].plot(Variable(X_test).data.numpy(), y_pred.data.numpy(), '-r')
    ax[0,0].scatter(Variable(X_train).data.numpy(), Variable(T_train).data.numpy(), c='b', marker='.')
    ax[0,0].set_ylim([-0.1,1.1])
    ax[0,0].set_title(r'$f(x)=x^{2}$')
    #Plot hidden units on different Y-scale
    ax0 = ax[0,0].twinx()
    ax0.plot(Variable(X_test.expand(X_test.size(0),3)).data.numpy(), y_hidden.data.numpy(),ls='--')
    ax0.set_yticks([])

    #Sine function
    X_train, T_train = generateTrainingData(N,0,func='sine')
    X_test, T_test = generateTrainingData(100,0,func='sine')
    sNN = simpleNN(1, 3, 1, lr)
    sNN.trainNN(X_train, T_train, err)

    y_pred = sNN.getTPred(X_test)
    y_hidden = sNN.getHiddenUnits(X_test)
    ax[0,1].plot(Variable(X_test).data.numpy(), y_pred.data.numpy(), '-r')
    ax[0,1].scatter(Variable(X_train).data.numpy(), Variable(T_train).data.numpy(), c='b', marker='.')
    ax[0,1].set_ylim([-1.1,1.1])
    ax[0,1].set_title(r'$f(x)=sin(x)$')
    #Plot hidden units on different Y-scale
    ax0 = ax[0,1].twinx()
    ax0.plot(Variable(X_test.expand(X_test.size(0),3)).data.numpy(), y_hidden.data.numpy(),ls='--')
    ax0.set_yticks([])

    #Abs function
    X_train, T_train = generateTrainingData(N,0,func='abs')
    X_test, T_test = generateTrainingData(100,0,func='abs')
    sNN = simpleNN(1, 3, 1, lr)
    sNN.trainNN(X_train, T_train, err)

    y_pred = sNN.getTPred(X_test)
    y_hidden = sNN.getHiddenUnits(X_test)
    ax[1,0].plot(Variable(X_test).data.numpy(), y_pred.data.numpy(), '-r')
    ax[1,0].scatter(Variable(X_train).data.numpy(), Variable(T_train).data.numpy(), c='b', marker='.')
    ax[1,0].set_ylim([-0.1,1.1])
    ax[1,0].set_title(r'$f(x)=\left| x \right|$')
    #Plot hidden units on different Y-scale
    ax0 = ax[1,0].twinx()
    ax0.plot(Variable(X_test.expand(X_test.size(0),3)).data.numpy(), y_hidden.data.numpy(),ls='--')
    ax0.set_yticks([])

    #Heaviside function
    X_train, T_train = generateTrainingData(N,0,func='heavy')
    X_test, T_test = generateTrainingData(100,0,func='heavy')
    sNN = simpleNN(1, 3, 1, lr)
    sNN.trainNN(X_train, T_train, err)

    y_pred = sNN.getTPred(X_test)
    y_hidden = sNN.getHiddenUnits(X_test)
    ax[1,1].plot(Variable(X_test).data.numpy(), y_pred.data.numpy(), '-r')
    ax[1,1].scatter(Variable(X_train).data.numpy(), Variable(T_train).data.numpy(), c='b', marker='.')
    ax[1,1].set_ylim([-0.1,1.1])
    ax[1,1].set_title(r'$f(x)=H(x)$')
    #Plot hidden units on different Y-scale
    ax0 = ax[1,1].twinx()
    ax0.plot(Variable(X_test.expand(X_test.size(0),3)).data.numpy(), y_hidden.data.numpy(),ls='--')
    ax0.set_yticks([])

    for (i,j), ax0 in np.ndenumerate(ax):
        ax0.set_xlim([-1.05,1.05])
        ax0.set_xticks([]) 
        ax0.set_yticks([]) 
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5, rect=[0,0, 1, 0.9])
    #plt.savefig('Figure5_03.png')
    plt.show()