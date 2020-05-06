import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

cuda = True if torch.cuda.is_available() else False
criterion = nn.CrossEntropyLoss()
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class LPOM():
    def __init__(self, layer):
        super().__init__()
        # Initial W
        self.W = []
        for i in range(len(layer)-1):
            weights = (torch.randn(layer[i+1],layer[i])-0.5)*2*np.sqrt(6/(layer[i]+layer[i+1]))
            bias = torch.zeros(layer[i+1],1)
            self.W.append(torch.cat((bias,weights), dim=1))
        self.layer = layer

    def fNN(self, trainX, trainY):
        # trainX : (# of feature , # of samples) images.T
        # trainY : (# of samples, # of classes) labels
        self.X = []
        self.X.append(trainX)
        n = len(self.W)+1
        for i in range(1,n):
            cons_x = torch.ones(1, self.X[i-1].shape[1])
            layer_x = torch.cat((cons_x, self.X[i-1]), dim=0)
            Y = self.W[i-1].mm(layer_x)
            self.X.append(F.relu(Y))
        # return normal size predicted Y
        self.pred_Y = self.X[-1].T
        self.loss = criterion(self.pred_Y, trainY)
    
    def block_training(self, trainX, trainY, mu):
        self.fNN(trainX, trainY)
        trainY = F.one_hot(trainY).type(FloatTensor)
        maxloop = 20
        num_samples = trainX.shape[1]
        n = len(self.X)
        L = 1
        lambd = 5
        cons_x = torch.ones(1, num_samples)

        for loop in range(maxloop):
            # update X
            for i in range(1,n):
                # update X_i
                if i != n-1: 
                    layer_x = torch.cat((cons_x, self.X[i-1]), dim=0)
                    term1 = self.W[i-1].mm(layer_x)
                    
                    layer_x2 = torch.cat((cons_x, self.X[i]), dim=0)
                    term2 = self.W[i][:,1:].T.mm(F.relu(self.W[i].mm(layer_x2)-self.X[i+1]))
                    self.X[i] = F.relu(term1 - mu[i+1]/mu[i]*term2)
                else:
                    layer_x = torch.cat((cons_x, self.X[n-2]), dim=0)
                    self.X[n-1] = F.relu(self.W[n-2].mm(layer_x)-(self.X[n-1]-trainY.T)/mu[n-1])
                
            # update W
            for i in range(n-1):
                layer_x = torch.cat((cons_x, self.X[i]), dim=0)
                if layer_x.shape[1] <= layer_x.shape[0]:
                    pinv = torch.inverse(layer_x.T.mm(layer_x)+lambd*torch.eye(num_samples)).mm(layer_x.T)
                else:
                    pinv = layer_x.T.mm(torch.inverse(layer_x.mm(layer_x.T)+lambd*torch.eye(layer_x.shape[0])))
                term1 = F.relu(self.W[i].mm(layer_x)-self.X[i+1]).mm(pinv)
                self.W[i] = self.W[i]-1/L*term1
    
                    

