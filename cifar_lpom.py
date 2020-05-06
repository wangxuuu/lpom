import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
from lpom_cpu import LPOM
import numpy as np

input_size = 3072       # The image size = 28 x 28 = 784
hidden_size = 300      # The number of nodes at the hidden layer
num_classes = 10       # The number of output classes. In this case, from 0 to 9
num_epochs = 60         # The number of times entire dataset is trained
batch_size = 100       # The size of input data took for one iteration


os.makedirs("./data/cifar", exist_ok=True)
os.makedirs("./result", exist_ok=True)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(root='./data/cifar', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True,)

test_dataset = datasets.CIFAR10(root='./data/cifar', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

hidden_layer = [300,100]
layer = [input_size, 4000, 1000, 4000, num_classes]
n = len(layer)
mu = 100*np.ones(n)
model = LPOM(layer)

def activ(X):
    type_ = 'relu'
    if type_ == 'relu':
        return np.maximum(0,X)

def feedforwardNN(W, X):
    X_flow = X
    n = len(W)+1
    for i in range(1,n):
        cons_x = np.ones((1, X.shape[1]))
        layer_x = np.concatenate((cons_x, X_flow), axis=0)
        X_flow = activ(W[i-1].dot(layer_x))
    return X_flow.T

def train_accuracy(weights, train_loader):
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.view(-1, 3072).numpy()
        labels = labels.numpy()
        outputs = feedforwardNN(weights, images.T)
        predicted = np.argmax(outputs, axis=1)  # Choose the best class from the output: The class with the best score
        total += labels.shape[0]                    # Increment the total count
        correct += (predicted == labels).sum()     # Increment the correct count
    acc = correct/total
    return acc

def test_accuracy(weights, test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 3072).numpy()
        labels = labels.numpy()
        outputs = feedforwardNN(weights, images.T)
        predicted = np.argmax(outputs, axis=1)  # Choose the best class from the output: The class with the best score
        total += labels.shape[0]                    # Increment the total count
        correct += (predicted == labels).sum()     # Increment the correct count
    acc = correct/total
    return acc

train_acc_list = []
test_acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
        images = images.view(-1, 3072).numpy()         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
        labels = F.one_hot(labels, num_classes=10).numpy()
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
plt.savefig('./lpom_cifar.png')
plt.show()

train_acc_cifar_lpom = np.array(train_acc_list)
np.save('./result/train_acc_cifar_lpom.npy', train_acc_cifar_lpom)
test_acc_cifar_lpom = np.array(test_acc_list)
np.save('./result/test_acc_cifar_lpom.npy', test_acc_cifar_lpom)