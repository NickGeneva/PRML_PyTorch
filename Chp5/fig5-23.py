'''
Nicholas Geneva
ngeneva@nd.edu
August 30, 2017
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
        self.f2 = th.nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the neural network
        Args:
            x (th.DoubleTensor): [N x D_in] column matrix of training inputs
        Returns:
            out (th.DoubleTensor): [N x D_out] matrix of neural network outputs
        """
        lin1 = self.f1(self.linear1(x))
        out = self.f2(self.linear2(lin1))
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

    def getOutputActivations(self, x):
        """
        Calculates and returns the output activations for the neural network
        Args:
            x (th.DoubleTensor): [N x D_in] column matrix of training inputs
        Returns:
            out (th.DoubleTensor): [N x D_out] matrix of neural network output activations
        """
        lin1 = self.f1(self.linear1(x))
        out = self.linear2(lin1)
        return out

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
        self.alpha = 0
        self.beta = 1

        self.loss_fn = th.nn.BCELoss(size_average=False)
        self.reg_fn = th.nn.MSELoss(size_average=False)
    
    def trainNNLikelyhood(self, x_train, y_train, err_limit):
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
            loss = self.getLoss(x_train, y_train) 
            #loss = self.loss_fn(y_pred, y_t)
            err = loss.data
            print(idx,err)

            # Zero the gradients before running the backward pass.
            self.model.zero_grad()
            loss.backward()

            for param in self.model.parameters():
                param.data -= self.lr * param.grad.data #Using Batch sharpest decent

            idx+=1
            if(idx > 5e4): #Give up after 1e6 attempts to train
                print('Interation break')
                break

        print('Training Loss for M='+str(self.H)+': '+str(loss.data.numpy()))
        print('Alpha: '+str(self.alpha))

    def priorTrainNN(self, x_train, y_train, err_limit):
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

        y_pred = self.model(x_t)

        loss = self.getLoss(x_train, y_train) 
        grad_params = th.autograd.grad(loss, self.model.parameters(), create_graph=True)
        #for grad in grad_params:
            #print(grad.size())
        
        hess = self.getHessian(x_train, y_train)
        e, v = th.eig(self.beta*hess.data, eigenvectors=False)
              
        d = 0
        for param in self.model.parameters():
            target = Variable(th.zeros(param.size())).type(dtype)
            d += self.reg_fn(param, target).data
        
        for i in range(20):
            gamma = th.sum(e[:,0]/(self.alpha + e[:,0]), 0)
            self.alpha = (gamma/d)[0]
            print(self.alpha)


        print('Alpha Updated to: '+str(self.alpha))
        #string = raw_input("Press Enter to continue...")

    def getHessian(self, x_train, t_train):
        """
        Calculates the refined posterior probabilities for a set of points
        (recommended batch to prevent recalc. of hessian)
        Args:
            x_test (th.DoubleTensor): [N x D_in] matrix of test set inputs to find posterior
        Returns:
            post (Variable): [W x W] matrix containing second order derivative
        """
        #First determine size of Hassian
        N = 0
        for param in self.model.parameters():
            N += sum(len(row) for row in param)
        hess = Variable(th.zeros((N,N))).type(dtype)
        hess_diag = Variable(th.zeros((N,N))).type(dtype)

        loss = self.getLoss2(x_train, t_train)
        grad_params0 = th.autograd.grad(loss, self.model.parameters(), create_graph=True)

        ni, nj = 0, 0
        for k, v in self.model.state_dict().iteritems():
            if('weight' in k.lower()):
                for i, row in enumerate(v):
                    for j, elem in enumerate(row):
                        eps = v[i,j]*1e-8
                        v[i,j] = v[i,j] - eps
                        #Compute loss
                        loss = self.getLoss2(x_train, t_train) 
                        #backprop
                        grad_params2 = th.autograd.grad(loss, self.model.parameters(), create_graph=True)
                        
                        v[i,j] = v[i,j] + 2*eps
                        #Compute loss
                        loss = self.getLoss2(x_train, t_train) 
                        #backprop
                        grad_params1 = th.autograd.grad(loss, self.model.parameters(), create_graph=True)
                        #Finite difference
                        grad_params = np.subtract(grad_params1, grad_params2)
                        #Flatten and store in hess
                        ni = 0
                        for grad in grad_params:
                            hess[ni:ni +grad.numel(), nj] = grad.view(grad.numel())/(2*eps)
                            ni = ni +grad.numel()
                        #Reset weight to original value
                        v[i,j] = v[i,j] - eps
                        nj += 1

            if('bias' in k.lower()):
                for i, elem in enumerate(v):
                    eps = v[i]*1e-8
                    v[i] = v[i] - eps
                    #Compute loss
                    loss = self.getLoss2(x_train, t_train) 
                    #backprop
                    grad_params2 = th.autograd.grad(loss, self.model.parameters(), create_graph=True)

                    v[i] = v[i] + 2*eps
                    #Compute loss
                    loss = self.getLoss2(x_train, t_train) 
                    #backprop
                    grad_params1 = th.autograd.grad(loss, self.model.parameters(), create_graph=True)
                    #Finite difference
                    grad_params = np.subtract(grad_params1, grad_params2)
                    #Flatten and store in hess
                    ni = 0
                    for grad in grad_params:
                        hess[ni:ni +grad.numel(), nj] = grad.view(grad.numel())/(2*eps)
                        ni = ni +grad.numel()
                    #Reset weight to original value
                    v[i] = v[i] - eps
                    nj += 1

        #print(hess)
        loss0 = self.getLoss2(x_train, t_train)
        
        #Full finite difference diag check
        ni = 0
        for k, v in self.model.state_dict().iteritems():
            if('weight' in k.lower()):
                for i, row in enumerate(v):
                    for j, elem in enumerate(row):
                        eps = v[i,j]*1e-5
                        v[i,j] = v[i,j] + 2*eps
                        #Compute loss
                        loss1 = self.getLoss2(x_train, t_train)
                        
                        v[i,j] = v[i,j] - 4*eps
                        #Compute loss
                        loss2 = self.getLoss2(x_train, t_train)

                        loss = 1/(4*(eps**2))*(loss1 - 2*loss0 + loss2)
                        v[i,j] = v[i,j] + 2*eps #Reset weight
                        
                        hess_diag[ni,ni] = loss
                        ni += 1
                        
            if('bias' in k.lower()):
                for i, elem in enumerate(v):
                    eps = v[i]*1e-5
                    v[i] = v[i] + 2*eps
                    #Compute loss
                    loss1 = self.getLoss2(x_train, t_train)
                    
                    v[i] = v[i] - 4*eps
                    #Compute loss
                    loss2 = self.getLoss2(x_train, t_train)

                    loss = 1/(4*(eps**2))*(loss1 - 2*loss0 + loss2)
                    v[i] = v[i] + 2*eps #Reset weight
                    
                    hess_diag[ni,ni] = loss
                    ni += 1
        #print(hess_diag)

        return hess

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

    def getOutputActivations(self, x_train):
        """
        Get the prediction of the NN for a given set of points
        Args:
            x_train (th.DoubleTensor): [N x D_in] matrix of training inputs
        Returns:
            a_out (Variable): [N x D_out]
        """
        x_t = Variable(x_train, requires_grad=False)
        a_out = self.model.getOutputActivations(x_t)
        return a_out

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
        x_t = Variable(x_test)
        y_t = Variable(t_test, requires_grad=False)

        y_pred = self.model(x_t) #__call__()
        #Now lets compute out loss
        p_pr = 0 #Prior
        for param in self.model.parameters():
            target = Variable(th.zeros(param.size())).type(dtype)
            p_pr += self.reg_fn(param, target)
        p_lh = self.loss_fn(y_pred.squeeze(), y_t)#Likelyhood
        loss = p_lh + (self.alpha/2.0)*p_pr
        return loss

    def getLoss2(self, x_test, t_test):
        """
        Get the prediction of the NN for a given set of points
        Args:
            x_test (th.DoubleTensor): [N x D_in] matrix of test set inputs
            t_test (th.DoubleTensor): [N x D_out] matrix of test set outputs
        Returns:
            loss (Variable): loss error for the given test set
        """
        x_t = Variable(x_test)
        y_t = Variable(t_test, requires_grad=False)

        y_pred = self.model(x_t) #__call__()
        #Now lets compute out loss
        loss = self.loss_fn(y_pred.squeeze(), y_t)#Likelyhood
        return loss

    def resetWeights(self):
        for x in self.model.modules():
            if isinstance(x, th.nn.Linear):
                x.weight.data = th.normal(means=th.zeros(x.weight.size())).type(dtype)
                x.bias.data = th.zeros(x.bias.size()).type(dtype)

    def getPosterior(self, x_train, t_train, x_test):
        """
        Calculates the refined posterior probabilities for a set of points
        (recommended batch to prevent recalc. of hessian)
        Args:
            x_test (th.DoubleTensor): [N x D_in] matrix of test set inputs to find posterior
        Returns:
            post (th.DoubleTensor): [N x D_out] matrix containing probabilites of the respective classes
        """
        #First crunch the hessian for out current training data
        hess = self.getHessian(x_train, t_train)
        A = self.alpha*th.eye(hess.size(0)).type(dtype) + self.beta*self.getHessian(x_train, t_train).data
        A_i = th.inverse(A)
        #Next lets get our gradient vectors
        N = 0
        for param in self.model.parameters():
            N += sum(len(row) for row in param)
        b = th.zeros((N,1)).type(dtype)
        #loss = self.getLoss(x_train, t_train)
        # Zero the gradients before running the backward pass.
        # self.model.zero_grad()
        # loss.backward()
        # i = 0
        # for param in self.model.parameters():
        #     b[i:i+param.grad.data.numel(), 0] =  param.grad.data.view(param.grad.data.numel())
        #     i = i + param.grad.data.numel()
        # #Now calculate sigma_a^2
        # sigma2 = th.mm(b.t(),th.inverse(A))
        # sigma2 = th.mm(sigma2, b)
        #Compute k(sigma2)
        #k = (1 + (np.pi*sigma2)/8)**(-0.5)
        k = Variable(th.DoubleTensor((x_test.size(0))))
        #Get the output activations a_out for all the test points
        a_out = self.model.getOutputActivations(Variable(x_test, requires_grad=False))
        for idx, a0 in enumerate(a_out):
            self.model.zero_grad()
            param_grads = th.autograd.grad(a0, self.model.parameters(), create_graph=True)
            #Find gradient of the output activation wrt the weights
            i = 0
            for grad0 in param_grads:
                b[i:i+grad0.data.numel(), 0] =  grad0.data.view(grad0.data.numel())
                i = i + grad0.data.numel()
            #Now calculate sigma_a^2
            sigma2 = th.mm(b.t(),A_i)
            sigma2 = th.mm(sigma2, b)
            #Compute k(sigma2)
            k[idx] = (1 + (np.pi*sigma2)/8)**(-0.5)
            
        func = th.nn.Sigmoid()
        return func(k.unsqueeze(1)*a_out).data

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
    f, ax = plt.subplots(1, 2, figsize=(8, 4))
    f.suptitle('Figure 5.23 pg. 284', fontsize=14)
    xlim, ylim = [-2.25,2.25], [-3,3]

    N = 50 #Number of points in each class
    X_train, T_train = generate2ClassData(N)
    D_in, H, D_out = 2, 8, 1
    lr, err = 5e-4, 1e-6 #learning rate, error threshold
    
    sNN = simpleNN(D_in, H, D_out, lr)
    sNN.trainNNLikelyhood(X_train, T_train[:,0], err)

    x = np.linspace(xlim[0],xlim[1],50)
    y = np.linspace(ylim[0],ylim[1],50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((X.shape[0],X.shape[1]))

    for i in range(3):
        sNN.priorTrainNN(X_train, T_train[:,0], err)
        sNN.trainNNLikelyhood(X_train, T_train[:,0], err)
    
    for (i,j), val in np.ndenumerate(X):
        x_test = th.DoubleTensor([[X[i,j], Y[i,j]]])
        Z[i,j] = sNN.getTPred(x_test).data.numpy()

    #Plot decision surface boundary
    ax[0].contour(X, Y, Z, levels = [0,0.1,0.3,0.5,0.7,0.9,1.0], cmap=plt.cm.brg, linewidth=0.5)

    x_test = th.DoubleTensor([X.flatten(), Y.flatten()]).t()
    x_out = sNN.getPosterior(X_train, T_train[:,0], x_test)

    Z = x_out.view(X.shape[0], X.shape[1])
    ax[1].contour(X, Y, Z.numpy(), levels = [0,0.1,0.3,0.5,0.7,0.9,1.0], cmap=plt.cm.brg, linewidth=0.5)

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
    
    for n, ax0 in np.ndenumerate(ax):
        ax0.scatter(X0[:,0], Y0[:,0], marker='o', facecolors='none', edgecolors='b')
        ax0.scatter(X0[:,1], Y0[:,1], c='r', marker='x')
        #plotIdealFit(ax0, xlim, ylim)
        
        ax0.set_xlim(xlim)
        ax0.set_xticks([-2, -1, 0, 1, 2]) 
        ax0.set_ylim(ylim)
        ax0.set_yticks([-2, -1, 0, 1, 2])

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5, rect=[0,0, 1, 0.9])
    #plt.savefig('Figure5_23.png')
    plt.show()