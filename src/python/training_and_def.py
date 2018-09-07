#!/usr/bin/env python2

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:35:43 2018

@author: samer

"""

#TODO: refactor into separate files for each network, and
#      separate files for nominal training versus special cases / debugging

from __future__ import print_function

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

from visdom import Visdom
from torch.autograd import Variable
from torchvision import datasets, transforms

import training_and_def_net3 as net3


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
# loss function (L-2 norm, cosine distance, etc.)
# 

#TODO: improve training data selection

#TODO: LIST OF EXPERIMENTS


#class Net(nn.Module): #NOTE: slightly modded LeNet
#  def __init__(self):
#    super(Net, self).__init__()
#    # 1 input image channel, 8 output channels, 11x11 square convolution kernel
#    # Conv2d(channels(layers, output channels, kernel size, stride, padding)
#    # kernel, stride, padding dimensions may be specified explicitly 
#    self.conv1 = nn.Conv2d(1, 8, 11, 1, 1)
#    self.conv2 = nn.Conv2d(8, 16, 5, 1, 1)
#    # an affine operation: y = Wx + b
#    self.fc1 = nn.Linear(16 * 30 * 30, 256)
#    self.fc2 = nn.Linear(256, 128)
#    self.fc3 = nn.Linear(128, 64)
#
#  def forward(self, x):
#    # Max pooling over a (4, 4) window 
#    x = F.max_pool2d(F.relu(self.conv1(x)), (4, 4))
#    x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
#    x = x.view(-1, self.num_flat_features(x))
#    x = F.relu(self.fc1(x))
#    x = F.relu(self.fc2(x))
#    x = self.fc3(x)
#    return x
#
#  def num_flat_features(self, x):
#    size = x.size()[1:]  # all dimensions except the batch dimension
#    num_features = 1
#    for s in size:
#      num_features *= s
#    return num_features


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

    #NOTE: BIG embeddings
    #self.fc3 = nn.Linear(128, 64)
    #self.fc4 = nn.Linear(64, 128)

    #NOTE: small embeddings
    self.fc3 = nn.Linear(128, 32)
    self.fc4 = nn.Linear(32, 128)

    self.fc5 = nn.Linear(128, 256)

  def forward(self, emb):
    x = F.relu(self.fc4(emb))
    x = self.fc5(x)

    return x

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

  def forward(self, emb):
    recreation = self.embeddingnet(emb)
    return recreation

############################## VISUALIZATION ##############################

class VisdomLinePlotter(object):
  # Plots to Visdom
  def __init__(self, env_name='main'):
    self.viz = Visdom()
    self.env = env_name
    self.plots = {}
  def plot(self, var_name, split_name, x, y):
    if var_name not in self.plots:
      self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
          legend=[split_name],
          title=var_name,
          xlabel='Epochs',
          ylabel=var_name
      ))
    else:
      self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

def visEmbeddingDecoder():
  embeddings = []
  infile = open('embeddings.txt', 'r')
  for line in infile:
    embedding_list = line.split()
    embedding = np.array(embedding_list, dtype=np.float32)
    embeddings.append(embedding)


  #TODO: interpolate embeddings


  model = Net3()
  model = model.cuda()
  tnet = TripletNet(model)
  tnet = tnet.cuda()

  resume = True
  checkpoint_to_load = 'model_checkpoints/current/most_recent.pth.tar'

  # optionally resume from a checkpoint
  if resume:
    if os.path.isfile(checkpoint_to_load):
      print("=> loading checkpoint '{}'".format(checkpoint_to_load))
      checkpoint = torch.load(checkpoint_to_load)
      tnet.load_state_dict(checkpoint['state_dict'])
      print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_to_load, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(checkpoint_to_load))

  n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
  print('  + Number of params: {}'.format(n_parameters))

  recreations = []
  for emb in embeddings:
    # turn training off
    tnet.eval()
    emb = Variable(torch.from_numpy(emb))
    emb = emb.cuda()
    recreation = tnet(emb)
    recreations.append(recreation)

  outfile = open('recreations.txt', 'a')
  for i in range(len(recreations)):
    recreation = recreations[i]
    recreation = recreation.cpu()
    recreation = recreation.data.numpy().tolist()
    for i in range(len(recreation)):
      outfile.write(str(recreation[i]))
      outfile.write(" ")
    outfile.write("\n")

  outfile.close();

######################### GENERATE EMBEDDINGS #########################

def generateEmbeddings(test_set, tnet):

  outfile = open('embeddings.txt', 'a')

  # switch to evaluation mode
  tnet.eval()

  data_id = 0

  #for batch_idx, (data1n, data2n, data3n, data1r, data2r, data3r) in enumerate(test_loader):
  test_set_len = len(test_set)
  #for idx in range(test_set_len):
  for idx in range(5):

    data1n, data1r = test_set.getSpecificItem(idx)
    data2n, data2r = test_set.getSpecificItem(0)
    data3n, data3r = test_set.getSpecificItem(0)

    data1n, data2n, data3n = Variable(data1n), Variable(data2n), Variable(data3n)
    data1r, data2r, data3r = Variable(data1r), Variable(data2r), Variable(data3r)
    data1n = data1n.cuda()
    data2n = data2n.cuda()
    data3n = data3n.cuda()
    data1r = data1r.cuda()
    data2r = data2r.cuda()
    data3r = data3r.cuda()

    # compute output
    # NOTE: reversing order of data because I'm not confident in changing down stream eval
    _, _, embedded_1, _, _, _ = tnet(data1n, data3n, data2n, data1r, data3r, data2r)

    # write embeddings to text file
    if data_id % 300 == 0:
      embedded_1 = embedded_1.cpu()
      embedded_1 = embedded_1.data.numpy().tolist()
      embedded_1 = embedded_1[0]
      for i in range(len(embedded_1)):
        outfile.write(str(embedded_1[i]))
        outfile.write(" ")
      outfile.write("\n")

    data_id = data_id + 1

  outfile.close();

############################## MAIN ##############################

def main():

  # load dataset
  #train_loader, test_loader = cptn.CurateTrainTest()
  test_set = cptn.SpecialTestSet()

  model = net3.Net3()
  model = model.cuda()
  tnet = net3.TripletNet(model)
  tnet = tnet.cuda()

  resume = True
  checkpoint_to_load = 'model_checkpoints/current/most_recent.pth.tar'

  # optionally resume from a checkpoint
  if resume:
    if os.path.isfile(checkpoint_to_load):
      print("=> loading checkpoint '{}'".format(checkpoint_to_load))
      checkpoint = torch.load(checkpoint_to_load)
      tnet.load_state_dict(checkpoint['state_dict'])
      print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_to_load, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(checkpoint_to_load))

  n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
  print('  + Number of params: {}'.format(n_parameters))

  # run dataset through network to get embeddings
  generateEmbeddings(test_set, tnet)

  # generate interpolated embeddings and save recreated scans from autoencoder
  visEmbeddingDecoder()

if __name__ == '__main__':
  main()
  print ("donzo")
