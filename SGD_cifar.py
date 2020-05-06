import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

cuda = True if torch.cuda.is_available() else False
# use GPU if available
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

input_size = 3072       # The image size = 28 x 28 = 784
hidden_size = 4000      # The number of nodes at the hidden layer
num_classes = 10       # The number of output classes. In this case, from 0 to 9
num_epochs = 100         # The number of times entire dataset is trained
batch_size = 100       # The size of input data took for one iteration
learning_rate = 0.001  # The speed of convergence
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

def accuracy_tes(net, test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 3072).type(FloatTensor))
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        total += labels.size(0)                    # Increment the total count
        correct += (predicted == labels).sum().item()     # Increment the correct count
    return correct / total

def accuracy_train(net, train_loader):
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = Variable(images.view(-1, 3072).type(FloatTensor))
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        total += labels.size(0)                    # Increment the total count
        correct += (predicted == labels).sum().item()     # Increment the correct count
    return correct / total


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()                    # Inherited from the parent class nn.Module
        self.layer1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.layer2 = nn.Linear(hidden_size, 1000)
        self.layer3 = nn.Linear(1000, hidden_size)                          # Non-Linear ReLU Layer: max(0,x)
        self.layer4 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)

    def forward(self, x):                              # Forward pass: stacking each layer together
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = self.layer4(out)
        return out

net = Net(input_size, hidden_size, num_classes)
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

train_acc_list = []
test_acc_list = []
print("="*10,"Training","="*10)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
        images = Variable(images.view(-1, 3072).type(FloatTensor))         # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28
        labels = Variable(labels)
        # images = images.view(-1, 28*28)        # Convert torch tensor to Variable: change image from a vector of size 784 to a matrix of 28 x 28

        optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        outputs = net(images)                             # Forward pass: compute the output class given a image
        loss = criterion(outputs, labels)                 # Compute the loss: difference between the output class and the pre-given label
        loss.backward()                                   # Backward pass: compute the weight
        optimizer.step()                                  # Optimizer: update the weights of hidden nodes

        if (i+1) % 100 == 0:                              # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
    train_acc = accuracy_train(net, train_loader)
    test_acc = accuracy_tes(net, test_loader)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

plt.figure()
plt.plot(train_acc_list, label='Train accuracy')
plt.plot(test_acc_list, label='Test accuracy')
plt.legend()
plt.savefig('./cifar_SGD.png')
plt.show()

train_acc_cifar_lpom = np.array(train_acc_list)
np.save('./result/train_acc_cifar_adam.npy', train_acc_cifar_lpom)
test_acc_cifar_lpom = np.array(test_acc_list)
np.save('./result/test_acc_cifar_adam.npy', test_acc_cifar_lpom)