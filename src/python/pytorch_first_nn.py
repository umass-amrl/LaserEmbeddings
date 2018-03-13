#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:35:43 2018

@author: samer

"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # Conv2d(channels(layers / depth), output channels, kernel size, stride, padding)
        # kernel, stride, padding dimensions may be specified explicitly 
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# tells you the whole network structure
net = Net()
print(net)

# tells you the structure of the parameters that can be tuned
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weightrint("hello")

# example of I/O to network. input must be 4D tensor: samples(images), channels, height, width
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

# compute MSE loss on the output
output = net(input)
target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# function gradients
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


# visualize graph of computation
from graphviz import Digraph
from torchviz import make_dot
#make_dot(loss).view() # I LIKE A YOU

# compute the gradient based on the loss
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

# does the actual gradient computation
loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# Naive way of updating weights
#learning_rate = 0.01
#for f in net.parameters():
#    f.data.sub_(f.grad.data * learning_rate)

# fancy way, supported by pytorch
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

print("donzo")
