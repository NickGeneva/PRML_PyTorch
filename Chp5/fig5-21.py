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
from torch.autograd import Function, Variable

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

# Inherit from Function
class MixDensity(Function):

    def __init__(self):
        super(MixDensity, self).__init__()

    # Note that both forward and backward are @staticmethods
    def forward(self, a_pi, a_mu, a_sigma, t):
        #Save inputs for backward prop
        self.save_for_backward(a_pi, a_mu, a_sigma, t)
        K = a_pi.size(1) #Number of components in mixture
        losses = th.zeros(K,t.size(0)).type(dtype)
        pi_t = th.sum(th.exp(a_pi),1)
        var = th.pow(th.exp(a_sigma),2)

        for i  in range(K):
            mix = th.exp(a_pi[:,i])/pi_t
            losses[i,:] = mix*(1.0/th.pow(2.0*np.pi*var[:,i],0.5))*th.exp(-0.5*th.pow(t-a_mu[:,i], 2)/var[:,i])
        
        loss = -th.sum(th.log(th.sum(losses,0)),0)
        return loss

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        a_pi, a_mu, a_sigma, t = self.saved_variables
        K = a_pi.size(1) #Number of components in mixture
        N = t.size(0) #Number of points in batch
        pi_grad, mu_grad, sigma_grad = None, None, None
        losses = Variable(th.zeros(K,t.size(0)).type(dtype))

        pi_t = th.sum(th.exp(a_pi),1)
        var = th.pow(th.exp(a_sigma),2)

        for i  in range(K):
            mix = th.exp(a_pi[:,i])/pi_t
            losses[i,:] = mix*(1.0/th.pow(2.0*np.pi*var[:,i],0.5))*th.exp(-0.5*th.pow(t-a_mu[:,i], 2)/var[:,i])
        # Gradient w.r.t. a_pi (Eq. 5.155)
        if(self.needs_input_grad[0]):
            pi_grad = th.zeros((N,K)).type(dtype)
            for i in range(K):
                gamma = losses[i,:]/th.sum(losses, 0)
                mix = th.exp(a_pi[:,i])/pi_t
                pi_grad[:,i] = (mix - gamma).data
        # Gradient w.r.t. a_mu (Eq. 5.156)
        if(self.needs_input_grad[1]):
            mu_grad = th.zeros((N,K)).type(dtype)
            for i in range(K):
                gamma = losses[i,:]/th.sum(losses, 0)
                mu_grad[:,i] = (gamma*(a_mu[:,i] - t)/var[:,i]).data
        # Gradient w.r.t. a_sigma (Eq. 5.157)
        if(self.needs_input_grad[2]):
            sigma_grad = th.zeros((N,K)).type(dtype)
            for i in range(K):
                gamma = losses[i,:]/th.sum(losses, 0)
                sigma_grad[:,i] = (gamma*(1 - th.pow(t - a_mu[:,i], 2)/var[:,i])).data

        return pi_grad, mu_grad, sigma_grad, None

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
        y_t = Variable(y_train.squeeze())
        idx = 0

        err0 = th.zeros(1).type(dtype)
        err = th.randn(1).type(dtype) + 1

        while (th.norm(err - err0) > err_limit):
            err0 = err
            err = th.randn(1).type(dtype)

            #Feed inputes into neural network
            
            hidden = self.model.hiddenForward(x_t).t()
            y_pred = self.model(x_t)

            self.loss_fn = MixDensity()
            a_pi = y_pred[:,0:3]
            a_mu = y_pred[:,3:6]
            a_sigma = y_pred[:,6:]
            loss = self.loss_fn(a_pi, a_mu, a_sigma, y_t)
            
            # Zero the gradients before running the backward pass.
            self.model.zero_grad()
            loss.backward(retain_graph=True)

            for param in self.model.parameters():
                #print('=========',param.grad)
                param.data -= self.lr * param.grad.data #Using Batch sharpest decent
            
            print(loss)
            #First calculate mixing coefficients
            # pi_t = th.exp(y_pred[:,0])+th.exp(y_pred[:,1])+th.exp(y_pred[:,2])
            # p1 = th.exp(y_pred[:,0])/pi_t
            # p2 = th.exp(y_pred[:,1])/pi_t
            # p3 = th.exp(y_pred[:,2])/pi_t

            # var = th.pow(th.exp(y_pred[:,6:]),2)

            # loss1 = (1.0/th.pow(2.0*np.pi*var[:,0],0.5))*th.exp(-0.5*th.pow(y_t-y_pred[:,3], 2)/var[:,0])
            # loss2 = (1.0/th.pow(2.0*np.pi*var[:,1],0.5))*th.exp(-0.5*th.pow(y_t-y_pred[:,4], 2)/var[:,1])
            # loss3 = (1.0/th.pow(2.0*np.pi*var[:,2],0.5))*th.exp(-0.5*th.pow(y_t-y_pred[:,5], 2)/var[:,2])

            # loss = -th.sum(th.log(p1*loss1 + p2*loss2 + p3*loss3),0)

            # print(loss, lossx)
            # gamma0 = p1*loss1 + p2*loss2 + p3*loss3
            # gamma1 = p1*loss1/gamma0
            # gamma2 = p2*loss2/gamma0
            # gamma3 = p3*loss3/gamma0

            # grads = Variable(th.zeros(9,5).type(dtype))
            # grads[0,:] = th.sum(hidden[:,:]*(p1 - gamma1),1)
            # grads[1,:] = th.sum(hidden[:,:]*(p2 - gamma2),1) 
            # grads[2,:] = th.sum(hidden[:,:]*(p3 - gamma3),1)

            # grads[3,:] = th.sum(hidden[:,:]*(gamma1*(y_pred[:,3]-y_t)/var[:,0]),1)
            # grads[4,:] = th.sum(hidden[:,:]*(gamma2*(y_pred[:,4]-y_t)/var[:,1]),1)
            # grads[5,:] = th.sum(hidden[:,:]*(gamma3*(y_pred[:,5]-y_t)/var[:,2]),1)

            # grads[6,:] = th.sum(hidden[:,:]*gamma1*(1 - th.pow(y_t - y_pred[:,3], 2)/var[:,0]),1)
            # grads[7,:] = th.sum(hidden[:,:]*gamma2*(1 - th.pow(y_t - y_pred[:,4], 2)/var[:,1]),1)
            # grads[8,:] = th.sum(hidden[:,:]*gamma3*(1 - th.pow(y_t - y_pred[:,5], 2)/var[:,2]),1)
            
            #Now lets compute out loss
            # err = loss.data

            # # Zero the gradients before running the backward pass.
            # self.model.zero_grad()
            # loss.backward()

            # for param in self.model.parameters():
            #     print(param.grad)
            #     param.data -= self.lr * param.grad.data #Using Batch sharpest decent

            idx+=1
            if(idx > 1e6): #Give up after 1e6 attempts to train
                print('Interation break')
                break

        print('Training Loss for M='+str(self.H)+': '+str(loss.data.numpy()))

    def getMixingCoeff(self, x_test):
        """
        Returns the set of mixing coefficients
        Args:
            x_test (th.DoubleTensor): [N x D_in] matrix of test inputs
        Returns:
            means (Variable[th.DoubleTensor]): [K x N] matrix of mixing coefficients for each of the supplied inputs
        """
        x_t = Variable(x_test, requires_grad=False)
        mix = Variable(th.zeros(3, x_test.size(0)).type(dtype))
        #Feed inputes into neural network
        y_pred = self.model(x_t)

        #First calculate mixing coefficients
        pi_t = th.exp(y_pred[:,0])+th.exp(y_pred[:,1])+th.exp(y_pred[:,2])
        mix[0,:] = th.exp(y_pred[:,0])/pi_t
        mix[1,:] = th.exp(y_pred[:,1])/pi_t
        mix[2,:] = th.exp(y_pred[:,2])/pi_t

        return mix

    def getMeans(self, x_test):
        """
        Returns the set of mixing coefficients
        Args:
            x_test (th.DoubleTensor): [N x D_in] matrix of test inputs
        Returns:
            mix (Variable[th.DoubleTensor]): [K x N] matrix of mixing coefficients for each of the supplied inputs
        """
        x_t = Variable(x_test, requires_grad=False)
        means = Variable(th.zeros(3,x_test.size(0)).type(dtype))
        #Feed inputes into neural network
        y_pred = self.model(x_t)

        #First calculate mixing coefficients
        means[0,:] = y_pred[:,3]
        means[1,:] = y_pred[:,4]
        means[2,:] = y_pred[:,5]

        return means

    def getTProbability(self, x_test, y_test):
        """
        Get the prediction of the NN for a given set of points
        Args:
            x_test (th.DoubleTensor): [N x D_in] matrix of test inputs
            y_test (th.DoubleTensor): [N x K] matrix of test outputs
        Returns:
            p (Variable[th.DoubleTensor]): [N]
        """
        x_t = Variable(x_test, requires_grad=False)
        y_t = Variable(y_test.squeeze(), requires_grad=False)
        #Feed inputes into neural network
        y_pred = self.model(x_t)
        #First calculate mixing coefficients
        pi_t = th.exp(y_pred[:,0])+th.exp(y_pred[:,1])+th.exp(y_pred[:,2])
        p1 = th.exp(y_pred[:,0])/pi_t
        p2 = th.exp(y_pred[:,1])/pi_t
        p3 = th.exp(y_pred[:,2])/pi_t

        #Compute Joint probability
        #Mixing coefficients
        pi_t = th.exp(y_pred[:,0])+th.exp(y_pred[:,1])+th.exp(y_pred[:,2])
        p1 = th.exp(y_pred[:,0])/pi_t
        p2 = th.exp(y_pred[:,1])/pi_t
        p3 = th.exp(y_pred[:,2])/pi_t
        #Variances
        var = th.pow(th.exp(y_pred[:,6:]),2)
        #Probabilities
        prob1 = (1.0/th.pow(2.0*np.pi*var[:,0],0.5))*th.exp(-0.5*th.pow(y_t-y_pred[:,3], 2)/var[:,0])
        prob2 = (1.0/th.pow(2.0*np.pi*var[:,1],0.5))*th.exp(-0.5*th.pow(y_t-y_pred[:,4], 2)/var[:,1])
        prob3 = (1.0/th.pow(2.0*np.pi*var[:,2],0.5))*th.exp(-0.5*th.pow(y_t-y_pred[:,5], 2)/var[:,2])

        prob = p1*prob1 + p2*prob2 + p3*prob3

        return prob

def generateData(L,N,std):
    """Generates a set of synthetic data evenly distributed along the X axis
        with a target function of X + 0.3sin(2*pi*X) with 
        uniform noise from [-0.1, 0.1] in the Y direction
    Args:
        L (Int) = Number of data sets desired
        N (Int) = Number of points in data set
        std (Array) = standard deviation of guassian noise
    Returns:
        X (th.DoubleTensor) = [N x L] matrix of X coords of target points
        T (th.DoubleTensor) = [N x L] matrix of Y coord of target points
    """
    X = th.linspace(0,1,N).unsqueeze(1).type(dtype)
    mu = X + 0.3*th.sin(2.0*np.pi*X).expand(N,L).type(dtype)
    if(std > 0):
        T = mu + (0.2*th.rand(N,L).type(dtype) - 0.1)
    else:
        T = mu
    return X,T

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    #Set up subplots
    f, ax = plt.subplots(2, 2, figsize=(8, 7))
    f.suptitle('Figure 5.21 pg. 276', fontsize=14)

    T_train, X_train = generateData(1,100,0.05)
    X_test, T_test = generateData(1,100,0)
    D_in, H, D_out = 1, 5, 9 #Dimension in, Hidden units, Dimension out 
    lr, err = 1e-6, 1e-5 #learning rate, error threshold

    sNN = simpleNN(D_in, H, D_out, lr)
    sNN.trainNN(X_train[:,0].unsqueeze(1), T_train[:,0].unsqueeze(1), err)

    colors = ['b', 'g', 'r']
    #Plot mixing coefficients
    mix = sNN.getMixingCoeff(X_test).data.numpy()
    for i in range(mix.shape[0]):
        ax[0,0].plot(Variable(X_test).data.numpy(), mix[i,:], c=colors[i])
    ax[0,0].set_ylabel(r'$\pi_{k}$')
    ax[0,0].set_title("Mixing Coefficients")
    #Plot means
    means = sNN.getMeans(X_test).data.numpy()
    for i in range(means.shape[0]):
        ax[0,1].plot(Variable(X_test).data.numpy(), means[i,:], c=colors[i])
    ax[0,1].set_ylabel(r'$\mu_{k}$')
    ax[0,1].set_title("Means")
    #Plot joint proabability
    p = 100
    X, Y = np.meshgrid(np.linspace(0,1,p), np.linspace(0,1,p))
    Z = np.zeros((p,p))

    for i, x0 in np.ndenumerate(np.linspace(0,1,p)):
        x = th.from_numpy(X[i,:]).t()
        y = th.from_numpy(Y[i,:]).t()
        Z[i,:] = sNN.getTProbability(x,y).data.numpy()
        
    ax[1,0].contour(X, Y, Z, 15, linewidths=1, cmap=plt.cm.gnuplot)
    ax[1,0].set_title("Conditional Mixture Probability")
    #Plot best fit based off means and max mixing coefficients
    ax[1,1].scatter(Variable(X_train[:,0]).data.numpy(), Variable(T_train[:,0]).data.numpy(), s=20, marker='o', facecolors='none', edgecolors='g')
    X = np.linspace(0,1,p)
    Y = np.zeros(p)
    for i, x0 in np.ndenumerate(np.linspace(0,1,p)):
        if (mix[0,i] > mix[1,i] and mix[0,i] > mix[2,i]):
            Y[i] = means[0,i]
        elif (mix[1,i] > mix[2,i]):
            Y[i] = means[1,i]
        else:
            Y[i] = means[2,i]
    #Break discontinuities by inserting Nan's
    pos = np.where(np.abs(np.diff(Y)) >= 0.05)[0] + 1
    X = np.insert(X, pos, np.nan)
    Y = np.insert(Y, pos, np.nan)
    ax[1,1].plot(X, Y, '-r')
    ax[1,1].set_title("Conditional Mode")

    for (i,j), ax0 in np.ndenumerate(ax):
        ax0.set_xlim([-0.1,1.1])
        ax0.set_xticks([0, 1]) 
        ax0.set_ylim([-0.1,1.1])
        ax0.set_yticks([0, 1]) 
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5, rect=[0,0, 1, 0.9])
    #plt.savefig('Figure5_21.png')
    plt.show()