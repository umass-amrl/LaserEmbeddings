#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:35:43 2018

@author: samer

"""


from __future__ import print_function
import torch

# unitialized tensor
x = torch.Tensor(5, 3)
print(x)

# randomly initialized tensor
x = torch.rand(5, 3)
print(x)

# size of tensor
print(x.size())

# add two tensors
y = torch.rand(5, 3)
print(x + y)

# add two tensors again, but differently
print(torch.add(x, y))

# add two tensors and store the result in 'result'
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

# add in place
# adds x to y
y.add_(x)
print(y)

# supports MATLAB style indexing
print(x[:, 1])

# 'view' function reshapes tensor. '-1' argument: "infer this dimension from the other dimensions"
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# vector of ones
a = torch.ones(5)
print(a)

# convert 'a' to a numpy array. store in 'b'.
b = a.numpy()
print(b)

# 'a' and 'b' SHARE MEMORY
a.add_(1)
print(a)
print(b)

# go back to torch tensor from numpy array
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# Use the 'Variable' class. 
# Variable class holds: data (tensor), gradients, gradients wrt to some function (grad_fn)
from torch.autograd import Variable

# Initialize an instance of the 'Variable' class. User-defined Variables do not have a grad_fn
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

# Create new variable through an operation on an existing variable. This variable has a grad_fn.
y = x + 2
print(y)

# 
print(y.grad_fn)

# 
z = y * y * 3
out = z.mean()
print(z, out)

# compute the gradient of 'out'. Equivalent to 'out.backward(torch.Tensor([1.0]))'
out.backward()

# print gradient
print(x.grad)

# autograd is god 
x = torch.randn(3)
#print(x)
#x = torch.ones(3)*0.01
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

#gradients = torch.FloatTensor([1.0, 1.0, 1.0])
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)

print("hello")
