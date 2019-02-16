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

############################## VARIATIONAL AUTOENCODER ##############################

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
    out = self.relu(self.conv1(x))
    out = self.relu(self.conv2(out))
    out = self.relu(self.conv3(out))
    out = self.relu(self.conv4(out))
    out = out.view(out.size(0), -1)
    h1 = self.relu(self.fc1(out))
    mu = self.fc21(h1)
    logvar = self.fc22(h1)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    if self.training:
      std = logvar.mul(0.5).exp_()
      eps = Variable(std.data.new(std.size()).normal_())
      return eps.mul(std).add_(mu)
    else:
      return mu

  def decode(self, z):
    h3 = self.relu(self.fc3(z))
    out = self.relu(self.fc4(h3))
    out = out.view(out.size(0), 8, 64, 64)
    out = self.relu(self.deconv1(out))
    out = self.relu(self.deconv2(out))
    out = self.relu(self.deconv3(out))
    out = self.ht(self.conv5(out))
    return out

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

############################## SIMILARITY NET ##############################

class SimNet(nn.Module):
  def __init__(self):
    super(SimNet, self).__init__()

    self.fc1 = nn.Linear(2 *  32, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, 1)

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, db_item_emb, query_emb):
     x = torch.cat(db_item_emb, query_emb)
     x = self.relu(self.fc1(x))
     x = self.relu(self.fc2(x))
     sim_score= self.sigmoid(self.fc3(x))
     return sim_score
