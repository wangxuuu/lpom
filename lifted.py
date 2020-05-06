import torch
import torch.nn as nn
import numpy as np

def activ(X):
    type_ = 'relu'
    if type_ == 'relu':
        return np.maximum(0,X)

class LIFT():
    def __init__(self, layer):
        super().__init__()
        # Initial W
        self.W = []
        for i in range(len(layer)-1):
            weights = (np.random.rand(layer[i+1],layer[i])-0.5)*2*np.sqrt(6/(layer[i]+layer[i+1]))
            bias = np.zeros((layer[i+1],1))
            self.W.append(np.concatenate((bias,weights), axis=1))
        self.layer = layer

    def fNN(self, trainX, trainY):
        # trainX : (# of feature , # of samples) images.T
        # trainY : (# of samples, # of classes) labels
        self.X = []
        self.X.append(trainX)
        n = len(self.W)+1
        for i in range(1,n):
            cons_x = np.ones((1, self.X[i-1].shape[1]))
            layer_x = np.concatenate((cons_x, self.X[i-1]), axis=0)
            Y = self.W[i-1].dot(layer_x)
            self.X.append(activ(Y))
        # return normal size predicted Y
        self.pred_Y = self.X[-1].T
        # print('sum_x: ', sum(sum(self.X[-1])))
        self.loss = sum(sum(np.power((self.X[-1].T-trainY),2)))/trainY.shape[0]
        # self.loss = criterion(self.pred_Y, trainY)
    
    def block_training(self, trainX, trainY, mu):
        self.fNN(trainX, trainY)
        maxloop = 20
        num_samples = trainX.shape[1]
        n = len(self.X)
        L = 1
        rho = 5
        lambd = 5
        cons_x = np.ones((1, num_samples))

        for _ in range(maxloop):
            # update X
            for i in range(1,n):
                # update X_i
                if i != n-1:
                    layer_x = np.concatenate((cons_x, self.X[i-1]), axis=0)
                    layer_x2 = np.concatenate((cons_x, self.X[i]), axis=0)
                    term1 = np.dot(self.W[i][:,:-1].T, self.W[i][:,:-1])+np.eye(self.W[i].shape[1]-1)
                    term2 = np.linalg.inv(term1).dot(self.W[i][:,:-1].T.dot(self.X[i+1]-self.W[i][:,-1].reshape(-1,1).dot(cons_x))+self.W[i-1].dot(layer_x))
                    self.X[i] = term2
                else:
                    self.X[n-1] = activ(1/(1+lambd)*trainY.T+lambd/(1+lambd)*self.X[-1])     
            # update W
            for i in range(n-1):
                layer_x = np.concatenate((cons_x, self.X[i]), axis=0)
                # self.W[i] = np.linalg.inv(np.dot(layer_x, layer_x.T) + rho/lambd * np.eye(layer_x.shape[0])).dot(
                #     layer_x).dot(self.X[i+1]).T
                term1 = np.concatenate((layer_x.T, np.sqrt(rho/lambd)*np.eye(layer_x.shape[0])),axis=0)
                term2 = self.W[i].T
                term3 = np.concatenate((self.X[i+1].T, np.zeros((layer_x.shape[0], self.X[i+1].shape[0]))), axis=0)

                self.W[i] = (term1.T.dot(np.linalg.inv(term1.dot(term1.T))).dot(term3)).T
