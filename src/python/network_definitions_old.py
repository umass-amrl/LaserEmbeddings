#!/usr/bin/env python2

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:35:43 2018

@author: samer

"""

#TODO: refactor into separate files for each network, and
#      separate files for nominal training versus special cases / debugging

from __future__ import print_function

import rospy
#from std_msgs.msg import string
from std_msgs.msg import String


import os
#import copy
import torch
import shutil
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import convert_png_to_numpy as cptn
import torch.backends.cudnn as cudnn

from scipy import spatial
from visdom import Visdom
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.decomposition import PCA, KernelPCA



#NOTE: EXPERIMENTAL PARAMS:

# input shape / type (1d, 2d flat, 2d perdiodic rotation)

# input size (downsampling rate)

# input richness (single scan vs history of scans)

# input noising (yes / no / how)

# network structure (num layers, convolution vs max pooling vs fully connected, etc.)

# dimensionality of output (embedding)

# gradient descent methods (SGD, ADOM, Momentum, etc)

# training hacks (dropout, batch size)

# hyperparameters (learning rate, biases, initialization)

# activation function (ReLU, tanh, sigmoid, etc.)

# training example selection (selection of pairs, shuffling, presentation order, etc.)
    # regular (time-based)
    # noised
    # objectified
# loss function (L-2 norm, cosine distance, etc.)

# 

#TODO: improve training data selection

#TODO: LIST OF EXPERIMENTS

#embeddings_database = []

#TODO: seperate out all NN class stuff to a different file

############################## NETWORK STRUCTURE ##############################





class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(8 * 64 * 64, 512)

        # Latent space
        self.fc21 = nn.Linear(512, 32)
        self.fc22 = nn.Linear(512, 32)

        # Decoder
        self.fc3 = nn.Linear(32, 512)
        self.fc4 = nn.Linear(512, 8 * 64 * 64)
        self.deconv1 = nn.ConvTranspose2d(8, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 8, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.ht = nn.Hardtanh()

    def encode(self, x):
        #print(x.shape)
        out = self.relu(self.conv1(x))
        #print(out.shape)
        out = self.relu(self.conv2(out))
        #print(out.shape)
        out = self.relu(self.conv3(out))
        #print(out.shape)
        out = self.relu(self.conv4(out))
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        h1 = self.relu(self.fc1(out))
        #print(h1.shape)
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        #print(mu.shape)
        #print(logvar.shape)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        #print(z.shape)
        h3 = self.relu(self.fc3(z))
        #print(h3.shape)
        out = self.relu(self.fc4(h3))
        #print(out.shape)
        # import pdb; pdb.set_trace()
        out = out.view(out.size(0), 8, 64, 64)
        #print(out.shape)
        out = self.relu(self.deconv1(out))
        #print(out.shape)
        out = self.relu(self.deconv2(out))
        #print(out.shape)
        out = self.relu(self.deconv3(out))
        #print(out.shape)
        #out = self.sigmoid(self.conv5(out))
        out = self.ht(self.conv5(out))
        #print(out.shape)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar











class ConvNet(nn.Module): #NOTE: 
  def __init__(self):
    super(ConvNet, self).__init__()

    # square kernel
    self.conv1 = nn.Conv2d(1, 8, (3, 3), stride=3, padding=1) #256 --> 86
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) #86 --> 44
    self.conv2 = nn.Conv2d(8, 16, (3, 3), stride=2, padding=1) #44 --> 22
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #22 --> 10
    self.conv3 = nn.Conv2d(16, 8, (3, 3), stride=2, padding=1) #10 --> 5
    self.conv4 = nn.Conv2d(8, 4, (3, 3), stride=1, padding=1) #5 --> 5
    

#    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#    self.conv4 = nn.Conv2d(8, 8, (3, 3), stride=1, padding=1) #32 --> 16
#    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#    self.conv5 = nn.Conv2d(8, 16, (3, 3), stride=1, padding=1) #16 --> 8
#    self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

#    self.fc1 = nn.Linear(16 * 8 * 8, 64)
#    self.fc2 = nn.Linear(256, 128)
#    self.fc3 = nn.Linear(128, 64)
#    self.embedlayer = nn.Linear(64, 32)
#    self.unembedlayer = nn.Linear(32, 64)
#    self.fc4 = nn.Linear(64, 128)
#    self.fc5 = nn.Linear(128, 256)
#    self.fc6 = nn.Linear(64, 16 * 8 * 8)

    #self.unpool5 = nn.MaxUnpool2d(kernel_size=2)
    self.deconv5 = nn.ConvTranspose2d(4, 8, kernel_size=3, stride=1, padding=1, output_padding=0) #5 --> 5
    #self.unpool4 = nn.MaxUnpool2d(kernel_size=2)
    self.deconv4 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1) #5 --> 10
    #self.unpool3 = nn.MaxUnpool2d(kernel_size=2)
    self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=3, padding=1, output_padding=0) #10 --> 44
    #self.unpool2 = nn.MaxUnpool2d(kernel_size=2)
    self.deconv2 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1, output_padding=0) #44 --> 256
    #self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
    self.deconv1 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1, output_padding=0)

#TODO: deconv


  def forward(self, x):

    #print(x.shape)
    x = F.relu(self.conv1(x))
    #print(x.shape)
    x, idx1 = self.pool1(x)
    #print(x.shape)
    x = F.relu(self.conv2(x))
    #print(x.shape)
    x, idx2 = self.pool2(x)
    #print(x.shape)
    x = F.relu(self.conv3(x))
    #print(x.shape)
    x, idx3 = self.pool3(x)
    #print(x.shape)
    x = F.relu(self.conv4(x))
    #print(x.shape)
    x, idx4 = self.pool4(x)
    #print(x.shape)
    x = F.relu(self.conv5(x))
    #print(x.shape)
    x, idx5 = self.pool5(x)
    #print(x.shape)

    
    x = x.view(-1, self.num_flat_features(x))
    #print(x.shape)

    x = F.relu(self.fc1(x))
    #print(x.shape)
#    x = F.relu(self.fc2(x))
    #print(x.shape)
#    x = F.relu(self.fc3(x))
    #print(x.shape)
    x = F.relu(self.embedlayer(x))
    #print(x.shape)
    emb_x = x

    x = F.relu(self.unembedlayer(x))
    #print(x.shape)
#    x = F.relu(self.fc4(x))
    #print(x.shape)
#    x = F.relu(self.fc5(x))
    #print(x.shape)
    x = F.relu(self.fc6(x))
    #print(x.shape)

    x = x.view(-1, 8, 8, 8)
    #x = x.view(-1, 8, 16, 16)
    #print(x.shape)

    #x = x.unsqueeze(1)
    #print(x.shape)
    

    x = self.unpool5(x, idx5)
    #print(x.shape)
    x = F.relu(self.deconv5(x))
    #print(x.shape)
    x = self.unpool4(x, idx4)
    #print(x.shape)
    x = F.relu(self.deconv4(x))
    #print(x.shape)
    x = self.unpool3(x, idx3)
    #print(x.shape)
    x = F.relu(self.deconv3(x))
    #print(x.shape)
    x = self.unpool2(x, idx2)
    #print(x.shape)
    x = F.relu(self.deconv2(x))
    #print(x.shape)
    x = self.unpool1(x, idx1)
    #print(x.shape)
    x = self.deconv1(x)
    #print(x.shape)

    x = F.hardtanh(x)
    #print(x.shape)

    #return x, emb_x
    return x, x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features


############################## NET NET ##############################







class DenoisingAutoEncoder(nn.Module):
    r"""
    Denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.
    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
    """
    def __init__(self, input_size, output_size):
        super(DenoisingAutoEncoder, self).__init__()

        self.forward_pass = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())
        self.backward_pass = nn.Sequential(nn.Linear(output_size, input_size), nn.Hardtanh())

        self.criterion = torch.nn.PairwiseDistance(p=1)
        #self.criterion = torch.nn.PairwiseDistance(p=2)
        #self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, x):
        # Train each autoencoder individually
#        x = x.detach()
        # Add noise, but use the original lossless input as the target.
#        x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.0002)) > -.1).type_as(x)
#        y = self.forward_pass(x_noisy)
        x = self.forward_pass(x)
        return x

#        if self.training:
#            x_reconstruct = self.backward_pass(y)
            #print(x.shape)
            #print(x_reconstruct.shape)
            #print(self.criterion(x_reconstruct, Variable(x.data, requires_grad=False)).shape)
            #print(x.cpu().data.numpy())
            #print(x_reconstruct.cpu().data.numpy())
#            loss = torch.mean(self.criterion(x_reconstruct, Variable(x.data, requires_grad=False)))
#            self.optimizer.zero_grad()
#            loss.backward()
#            self.optimizer.step()

#        return y.detach(), loss

    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):
    r"""
    A stacked autoencoder made from the denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = DenoisingAutoEncoder(256, 128)
        self.ae2 = DenoisingAutoEncoder(128, 64)
        self.ae3 = DenoisingAutoEncoder(64, 32)

        self.criterion = torch.nn.PairwiseDistance(p=1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x[:, 0, 0, :]

        xr = self.ae1(x)
        xr = self.ae2(xr)
        xr = self.ae3(xr)

#        a1, l1 = self.ae1(x)
#        a2, l2 = self.ae2(a1)
#        a3, l3 = self.ae3(a2)

        #total_loss = l1 + l2 + l3
#        total_loss = l1

#        if self.training:
#            return a3, total_loss

#        else:
#            return a3, self.reconstruct(a3)

        xr = self.ae3.reconstruct(xr)
        xr = self.ae2.reconstruct(xr)
        x_reconstruct = self.ae1.reconstruct(xr)


        loss = torch.mean(self.criterion(x_reconstruct, Variable(x.data, requires_grad=False)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #print(x.cpu().data.numpy())

        return x_reconstruct, loss

    def reconstruct(self, x):
        a2_reconstruct = self.ae3.reconstruct(x)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct











class NetnetDecoder(nn.Module):
  def __init__(self):
    super(Netnet, self).__init__()

    # non-square kernel
    #self.conv1_1 = nn.Conv1d(1, 1, 3, stride=1, padding=1)
    self.conv1_2 = nn.Conv1d(1, 1, 5, stride=1, padding=2)
    self.conv1_3 = nn.Conv1d(1, 1, 9, stride=1, padding=4)
    self.conv1_4 = nn.Conv1d(1, 1, 17, stride=1, padding=8)
    self.conv1_5 = nn.Conv1d(1, 1, 33, stride=1, padding=16)
    self.conv1_6 = nn.Conv1d(1, 1, 65, stride=1, padding=32)
    #self.conv1_7 = nn.Conv1d(1, 1, 129, stride=1, padding=64)

    self.bn1_1 = nn.BatchNorm2d(1)
    self.bn1_2 = nn.BatchNorm2d(1)
    self.bn1_3 = nn.BatchNorm2d(1)
    self.bn1_4 = nn.BatchNorm2d(1)
    self.bn1_5 = nn.BatchNorm2d(1)
    self.bn1_6 = nn.BatchNorm2d(1)
    self.bn1_7 = nn.BatchNorm2d(1)

    #self.fc1 = nn.Linear(7 * 256, 1024)
    self.fc1 = nn.Linear(5 * 256, 1024)
    self.bn_fc1 = nn.BatchNorm1d(1024)

    self.fc2 = nn.Linear(1024, 512)
    self.bn_fc2 = nn.BatchNorm1d(512)

    self.fc3 = nn.Linear(512, 256)
    self.bn_fc3 = nn.BatchNorm1d(256)

    self.fc4 = nn.Linear(256, 128)
    self.bn_fc4 = nn.BatchNorm1d(128)

    self.fc5 = nn.Linear(128, 64)
    self.bn_fc5 = nn.BatchNorm1d(64)

    self.fc6 = nn.Linear(64, 128)
    self.bn_fc6 = nn.BatchNorm1d(128)

    self.fc7 = nn.Linear(128, 256)
    self.bn_fc7 = nn.BatchNorm1d(256)

    self.fc8 = nn.Linear(256, 512)
    self.bn_fc8 = nn.BatchNorm1d(512)

    self.fc9 = nn.Linear(512, 1024)
    self.bn_fc9 = nn.BatchNorm1d(1024)

    self.fc10 = nn.Linear(1024, 5 * 256)


#    self.deconv1_2 = nn.ConvTranspose1d(1, 1, 5, stride=1, padding=2, output_padding=0)
#    self.deconv1_3 = nn.ConvTranspose1d(1, 1, 9, stride=1, padding=3, output_padding=0)
#    self.deconv1_4 = nn.ConvTranspose1d(1, 1, 17, stride=1, padding=8, output_padding=0)
#    self.deconv1_5 = nn.ConvTranspose1d(1, 1, 33, stride=1, padding=16, output_padding=0)
#    self.deconv1_6 = nn.ConvTranspose1d(1, 1, 65, stride=1, padding=32, output_padding=0)


    self.deconv1_2 = nn.ConvTranspose1d(1, 1, 5, stride=1, padding=2, output_padding=0)
    self.deconv1_3 = nn.ConvTranspose1d(1, 1, 9, stride=1, padding=4, output_padding=0)
    self.deconv1_4 = nn.ConvTranspose1d(1, 1, 17, stride=1, padding=8, output_padding=0)
    self.deconv1_5 = nn.ConvTranspose1d(1, 1, 33, stride=1, padding=16, output_padding=0)
    self.deconv1_6 = nn.ConvTranspose1d(1, 1, 65, stride=1, padding=32, output_padding=0)

  def forward(self, x):

    x = F.relu(self.fc6(x))
#    x = self.bn_fc6(x)
#    x = self.do_1d(x)

    x = F.relu(self.fc7(x))
#    x = self.bn_fc6(x)
#    x = self.do_1d(x)

    x = F.relu(self.fc8(x))
#    x = self.bn_fc6(x)
#    x = self.do_1d(x)

    x = F.relu(self.fc9(x))
#    x = self.bn_fc6(x)
#    x = self.do_1d(x)

    x = F.relu(self.fc10(x))
#    x = self.bn_fc6(x)
#    x = self.do_1d(x)

    x = x.unsqueeze(1)

    sec2 = x[:, :, 0:256]
    sec3 = x[:, :, 256:512]
    sec4 = x[:, :, 512:768]
    sec5 = x[:, :, 768:1024]
    sec6 = x[:, :, 1024:1280]

    sec2 = F.relu(self.deconv1_2(sec2))
    sec3 = F.relu(self.deconv1_3(sec3))
    sec4 = F.relu(self.deconv1_4(sec4))
    sec5 = F.relu(self.deconv1_5(sec5))
    sec6 = F.relu(self.deconv1_6(sec6))

    x = torch.tanh(sec2 + sec3 + sec4 + sec5 + sec6)

    return x

class Net3(nn.Module): #NOTE: 
  def __init__(self):
    super(Net3, self).__init__()

    # non-square kernel 
    self.conv1_1 = nn.Conv2d(1, 1, (1, 3), 1, (0, 1))
    self.conv1_2 = nn.Conv2d(1, 1, (1, 5), 1, (0, 2))
    self.conv1_3 = nn.Conv2d(1, 1, (1, 9), 1, (0, 4))
    self.conv1_4 = nn.Conv2d(1, 1, (1, 17), 1, (0, 8))
    self.conv1_5 = nn.Conv2d(1, 1, (1, 33), 1, (0, 16))
    self.conv1_6 = nn.Conv2d(1, 1, (1, 65), 1, (0, 32))
    self.conv1_7 = nn.Conv2d(1, 1, (1, 129), 1, (0, 64))
    self.conv2 = nn.Conv2d(14, 8, 11, 1, 1)
    self.conv3 = nn.Conv2d(8, 16, 5, 1, 1)
    self.fc1 = nn.Linear(16 * 30 * 30, 256)


    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 64)
#    self.fc4 = nn.Linear(64, 128)
#    self.fc5 = nn.Linear(64, 128)
    self.fc6 = nn.Linear(64, 128)
    self.fc7 = nn.Linear(128, 256)
#    self.fc8 = nn.Linear(128, 256)
#    self.fc9 = nn.Linear(256, 256)



  #def forward(self, x, rand_row):
  #  x = x[:,:,rand_row,:]
  def forward(self, x):
    #x = x[:,0, 0,:]
    x = x[:,:, 0,:]
    
    #print(emb.shape)
    x = F.relu(self.fc2(x))
#    print(x.shape)
    x = F.relu(self.fc3(x))
#    print(x.shape)
#    x = F.relu(self.fc4(x))
#    x = F.relu(self.fc5(x))
#    print(x.shape)
    x = F.relu(self.fc6(x))
#    print(x.shape)
#    x = F.relu(self.fc7(x))
#    x = F.relu(self.fc8(x))
    #print(x.shape)
#    x = torch.tanh(self.fc9(x))
#    x = torch.tanh(self.fc8(x))
    x = torch.tanh(self.fc7(x))

    return x, x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features


############################## NET NET ##############################

class Netnet(nn.Module):
  def __init__(self):
    super(Netnet, self).__init__()

    # non-square kernel
    self.conv1_1 = nn.Conv1d(1, 1, 3, stride=1, padding=1)
    self.conv1_2 = nn.Conv1d(1, 1, 5, stride=1, padding=2)
    self.conv1_3 = nn.Conv1d(1, 1, 9, stride=1, padding=4)
    self.conv1_4 = nn.Conv1d(1, 1, 17, stride=1, padding=8)
    self.conv1_5 = nn.Conv1d(1, 1, 33, stride=1, padding=16)
    self.conv1_6 = nn.Conv1d(1, 1, 65, stride=1, padding=32)
    #self.conv1_7 = nn.Conv1d(1, 1, 129, stride=1, padding=64)

    self.bn1_1 = nn.BatchNorm2d(1)
    self.bn1_2 = nn.BatchNorm2d(1)
    self.bn1_3 = nn.BatchNorm2d(1)
    self.bn1_4 = nn.BatchNorm2d(1)
    self.bn1_5 = nn.BatchNorm2d(1)
    self.bn1_6 = nn.BatchNorm2d(1)
    self.bn1_7 = nn.BatchNorm2d(1)

    #self.fc1 = nn.Linear(7 * 256, 1024)
    self.fc1 = nn.Linear(6 * 256, 1024)
    #self.fc1 = nn.Linear(5 * 256, 1024)
    self.bn_fc1 = nn.BatchNorm1d(1024)

    self.fc2 = nn.Linear(1024, 512)
    self.bn_fc2 = nn.BatchNorm1d(512)

    self.fc3 = nn.Linear(512, 256)
    self.bn_fc3 = nn.BatchNorm1d(256)

    self.fc4 = nn.Linear(256, 128)
    self.bn_fc4 = nn.BatchNorm1d(128)

    self.fc5 = nn.Linear(128, 64)
    self.bn_fc5 = nn.BatchNorm1d(64)

    self.fc6 = nn.Linear(64, 128)
    self.bn_fc6 = nn.BatchNorm1d(128)

    self.fc7 = nn.Linear(128, 256)
    self.bn_fc7 = nn.BatchNorm1d(256)

    self.fc8 = nn.Linear(256, 512)
    self.bn_fc8 = nn.BatchNorm1d(512)

    self.fc9 = nn.Linear(512, 1024)
    self.bn_fc9 = nn.BatchNorm1d(1024)

    #self.fc10 = nn.Linear(1024, 5 * 256)
    #self.fc10 = nn.Linear(1024, 6 * 256)
    self.fc10 = nn.Linear(1024, 256)

#    self.deconv1_2 = nn.ConvTranspose1d(1, 1, 5, stride=1, padding=2, output_padding=0)
#    self.deconv1_3 = nn.ConvTranspose1d(1, 1, 9, stride=1, padding=3, output_padding=0)
#    self.deconv1_4 = nn.ConvTranspose1d(1, 1, 17, stride=1, padding=8, output_padding=0)
#    self.deconv1_5 = nn.ConvTranspose1d(1, 1, 33, stride=1, padding=16, output_padding=0)
#    self.deconv1_6 = nn.ConvTranspose1d(1, 1, 65, stride=1, padding=32, output_padding=0)


    self.deconv1_1 = nn.ConvTranspose1d(1, 1, 3, stride=1, padding=1, output_padding=0)
    self.deconv1_2 = nn.ConvTranspose1d(1, 1, 5, stride=1, padding=2, output_padding=0)
    self.deconv1_3 = nn.ConvTranspose1d(1, 1, 9, stride=1, padding=4, output_padding=0)
    self.deconv1_4 = nn.ConvTranspose1d(1, 1, 17, stride=1, padding=8, output_padding=0)
    self.deconv1_5 = nn.ConvTranspose1d(1, 1, 33, stride=1, padding=16, output_padding=0)
    self.deconv1_6 = nn.ConvTranspose1d(1, 1, 65, stride=1, padding=32, output_padding=0)

  def forward(self, xn, rand_row):

    xn = xn[:,:,rand_row,:]

#    print("TOPTOPTOPTOTPOTP")
#    print(xn[0, 0, :])
    #print(xn.shape)
    #print(rand_row)
    #print("TOP/BOT")

    xn1 = F.relu(self.conv1_1(xn))
    xn2 = F.relu(self.conv1_2(xn))
    xn3 = F.relu(self.conv1_3(xn))
    xn4 = F.relu(self.conv1_4(xn))
    xn5 = F.relu(self.conv1_5(xn))
    xn6 = F.relu(self.conv1_6(xn))
#    xn7 = F.relu(self.conv1_7(xn))

#    bnxn1 = self.bn1_1(xn1)
#    bnxn2 = self.bn1_2(xn2)
#    bnxn3 = self.bn1_3(xn3)
#    bnxn4 = self.bn1_4(xn4)
#    bnxn5 = self.bn1_5(xn5)
#    bnxn6 = self.bn1_6(xn6)
#    bnxn7 = self.bn1_7(xn7)

#    outputs = [xn1, xn2, xn3, xn4, xn5, xn6, xn7]
    outputs = [xn1, xn2, xn3, xn4, xn5, xn6]
#    outputs = [xn2, xn3, xn4, xn5, xn6]

    x = torch.cat(outputs, 2) #7 x 256 = 1792
#    print(x.shape)

    x = x.view(-1, self.num_flat_features(x))
#    print(x.shape)

    x = F.relu(self.fc1(x))
    x = self.bn_fc1(x)
 #   x = self.do_1d(x)

    x = F.relu(self.fc2(x))
    x = self.bn_fc2(x)
#    x = self.do_1d(x)

    x = F.relu(self.fc3(x))
    x = self.bn_fc3(x)
#    x = self.do_1d(x)

    x = F.relu(self.fc4(x))
    x = self.bn_fc4(x)
#    x = self.do_1d(x)
    emb_n1 = x

    x = F.relu(self.fc5(x))
    x = self.bn_fc5(x)
#    x = self.do_1d(x)
    emb_0 = x

    x = F.relu(self.fc6(x))
    x = self.bn_fc6(x)
#    x = self.do_1d(x)
    emb_p1 = x

#    x = torch.tanh(self.fc7(x))

    x = F.relu(self.fc7(x))
    x = self.bn_fc7(x)
#    x = self.do_1d(x)

    """
    x = F.relu(self.fc4(xn))
#    x = self.bn_fc4(x)
#    x = self.do_1d(x)
    emb_n1 = x

    x = F.relu(self.fc5(x))
#    x = self.bn_fc5(x)
#    x = self.do_1d(x)
    emb_0 = x

    x = F.relu(self.fc6(x))
#    x = self.bn_fc6(x)
#    x = self.do_1d(x)
    emb_p1 = x

#    x = torch.tanh(self.fc7(x))

#    x = F.relu(self.fc7(x))
    x = torch.tanh(self.fc7(x))
#    x = self.bn_fc7(x)
#    x = self.do_1d(x)
    """
    
    x = F.relu(self.fc8(x))
    x = self.bn_fc8(x)
#    x = self.do_1d(x)

    x = F.relu(self.fc9(x))
    x = self.bn_fc9(x)
#    x = self.do_1d(x)

#    x = F.relu(self.fc10(x))
    x = torch.tanh(self.fc10(x))
#    x = self.bn_fc6(x)
#    x = self.do_1d(x)

#    x = x.unsqueeze(1)

#    sec1 = x[:, :, 0:256]
#    sec2 = x[:, :, 256:512]
#    sec3 = x[:, :, 512:768]
#    sec4 = x[:, :, 768:1024]
#    sec5 = x[:, :, 1024:1280]
#    sec6 = x[:, :, 1280:1536]

#    sec1 = F.relu(self.deconv1_1(sec1))
#    sec2 = F.relu(self.deconv1_2(sec2))
#    sec3 = F.relu(self.deconv1_3(sec3))
#    sec4 = F.relu(self.deconv1_4(sec4))
#    sec5 = F.relu(self.deconv1_5(sec5))
#    sec6 = F.relu(self.deconv1_6(sec6))

    #print(sec2.shape)
    #print(sec3.shape)
    #print(sec4.shape)
    #print(sec5.shape)
    #print(sec6.shape)

    #x = torch.tanh(sec2 + sec3 + sec4 + sec5 + sec6)
#    x = torch.tanh(sec1 + sec2 + sec3 + sec4 + sec5 + sec6)
    #x = sec2 + sec3 + sec4 + sec5 + sec6

#    x = torch.tanh(self.fc9(x))

    return emb_n1, emb_p1, emb_0, x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

############################## TRIPLET NET ##############################

class TripletNet(nn.Module):
  def __init__(self, embeddingnet):
    super(TripletNet, self).__init__()
    self.embeddingnet = embeddingnet

    #self.fc11 = nn.Linear(3 * (2 * 32), 64)
    self.fc11 = nn.Linear(2 * 64, 64)

    self.fc12 = nn.Linear(64, 32)

    self.fc21 = nn.Linear(2 * 64, 64)

    self.fc22 = nn.Linear(64, 32)

    self.fc30 = nn.Linear(2 * 32, 32)

    self.fc31 = nn.Linear(32, 2)

  def forward(self, xn, yn, zn, xr, yr, zr, rr):
    rec_x, mu_x, logvar_x = self.embeddingnet(xn)
    #rec_y, mu_y, logvar_y = self.embeddingnet(yn)
    #rec_z, mu_z, logvar_z = self.embeddingnet(zn)





    #print(rec_x.shape)
#    print(rec_x[0, 0, :])
    #print(rec_x.data.numpy())
#    print("BOTBOTOBTOBTOOBTOBT")
    loss = rec_x
    return rec_x, mu_x, logvar_x, rec_x, loss, loss, loss, loss, loss, loss, loss, loss, 0.0
   #return emb_n1x, emb_p1x, emb_0x, rec_x, emb_n1x, emb_p1x, emb_0x, rec_x, emb_n1x, emb_p1x, emb_0x, rec_x, 0.0
"""
  def forward(self, xn, yn, zn, xr, yr, zr, rr):
    emb_n1x, emb_p1x, emb_0x, rec_x = self.embeddingnet(xr, rr)
    emb_n1y, emb_p1y, emb_0y, rec_y = self.embeddingnet(yn, 0)
    emb_n1z, emb_p1z, emb_0z, rec_z = self.embeddingnet(zn, 0)


    A = [emb_0x, emb_0y]
    #print(emb_0x.shape)

    xa = torch.cat(A, 1)

    xa = F.relu(self.fc11(xa))
    xa = F.relu(self.fc12(xa))


    B = [emb_0z, emb_0y]
    #print(emb_0x.shape)

    xb = torch.cat(B, 1)

    xb = F.relu(self.fc21(xb))
    xb = F.relu(self.fc22(xb))


    C = [xa, xb]
    xc = torch.cat(C, 1)

    xc = F.relu(self.fc30(xc))
    #sim = torch.sigmoid(self.fc31(xc))
    sim = torch.softmax(self.fc31(xc), dim=1)

    return emb_n1x, emb_p1x, emb_0x, rec_x, emb_n1y, emb_p1y, emb_0y, rec_y, emb_n1z, emb_p1z, emb_0z, rec_z, sim
"""

"""
class TripletNet(nn.Module):
  def __init__(self, embeddingnet):
    super(TripletNet, self).__init__()
    self.embeddingnet = embeddingnet
"""



############################## TRIPLET NET ##############################

"""
class TripletNet(nn.Module):
  def __init__(self, embeddingnet):
    super(TripletNet, self).__init__()
    self.embeddingnet = embeddingnet

  def forward(self, emb):
    recreation = self.embeddingnet(emb)
    return recreation
"""
