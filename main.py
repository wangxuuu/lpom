import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
from lpom import LPOM
import numpy as np

cuda = True if torch.cuda.is_available() else False
# use GPU if available
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

input_size = 784       # The image size = 28 x 28 = 784
hidden_size = 300      # The number of nodes at the hidden layer
num_classes = 10       # The number of output classes. In this case, from 0 to 9
num_epochs = 20         # The number of times entire dataset is trained
batch_size = 100       # The size of input data took for one iteration


os.makedirs("./data/mnist", exist_ok=True)
os.makedirs("./result", exist_ok=True)
train_dataset = datasets.MNIST(root='./data/mnist',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)

test_dataset = datasets.MNIST(root='./data/mnist',
                           train=False,
                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

hidden_layer = [300,100]
layer = [input_size, 300, 100, num_classes]
n = len(layer)
mu = 2*torch.ones(n)
model = LPOM(layer)

def feedforwardNN(W, X):
    X_flow = X
    n = len(W)+1
    for i in range(1,n):
        cons_x = torch.ones(1, X.shape[1])
        layer_x = torch.cat((cons_x, X_flow), dim=0)
        X_flow = F.relu(W[i-1].mm(layer_x))
    return X_flow.T

def train_accuracy(weights, train_loader):
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = Variable(images.view(-1, 28*28).type(FloatTensor))
        outputs = feedforwardNN(weights, images.T)
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        total += labels.size(0)                    # Increment the total count
        correct += (predicted == labels).sum()     # Increment the correct count
    acc = correct.item()/total
    return acc

def test_accuracy(weights, test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28).type(FloatTensor))
        outputs = feedforwardNN(weights, images.T)
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        total += labels.size(0)                    # Increment the total count
        correct += (predicted == labels).sum()     # Increment the correct count
    acc = correct.item()/total
    return acc

train_acc_list = []
test_acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
        images = Variable(images.view(-1, 28*28).type(FloatTensor))         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
        labels = Variable(labels)
        
        model.block_training(trainX=images.T, trainY=labels, mu=mu)
        model.fNN(trainX=images.T, trainY=labels)
        loss = model.loss
        model_weights = model.W
        if (i+1) % 100 == 0:                              # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss))
        
        train_acc = train_accuracy(model_weights, train_loader)
        test_acc = test_accuracy(model_weights, test_loader)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

plt.figure()
plt.plot(train_acc_list, label='Train accuracy')
plt.plot(test_acc_list, label='Test accuracy')
plt.legend()
plt.savefig('./lpom.png')
plt.show()