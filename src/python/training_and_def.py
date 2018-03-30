#!/usr/bin/env python2

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:35:43 2018

@author: samer

"""

from __future__ import print_function
import argparse
import os
import shutil
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import convert_png_to_numpy as cptn
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from torchvision import datasets, transforms
from visdom import Visdom



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
# loss function parameters
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

class Net(nn.Module): #NOTE: 1D conv w/ XXX
  def __init__(self):
    super(Net, self).__init__()
    # 1 input image channel, 8 output channels, 11x11 square convolution kernel
    # Conv2d(channels(layers, output channels, kernel size, stride, padding)
    # kernel, stride, padding dimensions may be specified explicitly 
    
    #non-square kernel 
    self.conv1_1 = nn.Conv2d(1, 1, (1, 3), 1, 1)
    self.conv1_2 = nn.Conv2d(1, 1, (1, 5), 1, 2)
    self.conv1_3 = nn.Conv2d(1, 1, (1, 9), 1, 4)
    self.conv1_4 = nn.Conv2d(1, 1, (1, 17), 1, 8)
    self.conv1_5 = nn.Conv2d(1, 1, (1, 33), 1, 16)
    self.conv1_6 = nn.Conv2d(1, 1, (1, 65), 1, 32)
    self.conv1_7 = nn.Conv2d(1, 1, (1, 129), 1, 64)

    #NOTE: Arch 1
    
    #NOTE: Arch 2
    
    #NOTE: Arch 3

    
    #NOTE: BIG embeddings

    #NOTE: small embeddings


    #self.conv2 = nn.Conv2d(8, 16, 5, 1, 1)
    # an affine operation: y = Wx + b
    #self.fc1 = nn.Linear(16 * 30 * 30, 256)
    #self.fc2 = nn.Linear(256, 128)
    #self.fc3 = nn.Linear(128, 64)

  def forward(self, xn, xr):
    xn1 = F.relu(self.conv1_1(xn))
    xr1 = F.relu(self.conv1_1(xr))
    xn2 = F.relu(self.conv1_2(xn))
    xr2 = F.relu(self.conv1_2(xr))
    xn3 = F.relu(self.conv1_3(xn))
    xr3 = F.relu(self.conv1_3(xr))
    xn4 = F.relu(self.conv1_4(xn))
    xr4 = F.relu(self.conv1_4(xr))
    xn5 = F.relu(self.conv1_5(xn))
    xr5 = F.relu(self.conv1_5(xr))
    xn6 = F.relu(self.conv1_6(xn))
    xr6 = F.relu(self.conv1_6(xr))
    xn7 = F.relu(self.conv1_7(xn))
    xr7 = F.relu(self.conv1_7(xr))


    
    #NOTE: Arch 1
    outputs = [xn1, xr1, xn2, xr2, xn3, xr3, xn4, xr4, xn5, xr5, xn6, xr6, xn7, xr7]
    x = torch.cat(outputs, 1)
    
    #NOTE: Arch 2
    
    #NOTE: Arch 3
    outputs = [xn1, xr1, xn2, xr2, xn3, xr3, xn4, xr4, xn5, xr5, xn6, xr6, xn7, xr7]
    x = torch.cat(outputs, 1)

    # Max pooling over a (4, 4) window 
    #x = F.max_pool2d(F.relu(self.conv1(x)), (4, 4))
    #x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
    #x = x.view(-1, self.num_flat_features(x))
    #x = F.relu(self.fc1(x))
    #x = F.relu(self.fc2(x))
    #x = self.fc3(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

class TripletNet(nn.Module):
  def __init__(self, embeddingnet):
    super(TripletNet, self).__init__()
    self.embeddingnet = embeddingnet

  def forward(self, x, y, z):
    embedded_x = self.embeddingnet(x)
    embedded_y = self.embeddingnet(y)
    embedded_z = self.embeddingnet(z)
    dist_a = F.pairwise_distance(embedded_x, embedded_y, 2) # L-2 norm
    dist_b = F.pairwise_distance(embedded_x, embedded_z, 2) # L-2 norm
    return dist_a, dist_b, embedded_x, embedded_y, embedded_z

class AverageMeter(object):
  # Computes and stores the average and current value
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

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


def train(train_loader, tnet, criterion, optimizer, epoch):
  losses = AverageMeter()
  accs = AverageMeter()
  emb_norms = AverageMeter()

  # switch to train mode
  tnet.train()
  for batch_idx, (data1, data2, data3) in enumerate(train_loader):
    data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)
    data1 = data1.cuda()
    data2 = data2.cuda()
    data3 = data3.cuda()

    # compute output
    # NOTE: reversing order of data because I'm not confident in changing down stream eval
    dista, distb, embedded_x, embedded_y, embedded_z = tnet(data1, data3, data2)
    # 1 means, dista should be larger than distb
    target = torch.FloatTensor(dista.size()).fill_(1)
    target = Variable(target)
    target = target.cuda()

    loss_triplet = criterion(dista, distb, target)
    loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
    loss = loss_triplet + 0.001 * loss_embedd

    # measure accuracy and record loss
    acc = accuracy(dista, distb)
    losses.update(loss_triplet.data[0], data1.size(0))
    accs.update(acc, data1.size(0))
    emb_norms.update(loss_embedd.data[0]/3, data1.size(0))

    # compute gradient and do optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    log_interval = 10

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{}]\t'
            'Loss: {:.4f} ({:.4f}) \t'
            'Acc: {:.2f}% ({:.2f}%) \t'
            'Emb_Norm: {:.2f} ({:.2f})'.format(
          epoch, batch_idx * len(data1), len(train_loader.dataset),
          losses.val, losses.avg, 
          100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
  # log avg values to somewhere
  plotter.plot('acc', 'train', epoch, accs.avg)
  plotter.plot('loss', 'train', epoch, losses.avg)
  plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)


def test(test_loader, tnet, criterion, epoch):
  losses = AverageMeter()
  accs = AverageMeter()

  # switch to evaluation mode
  tnet.eval()
  for batch_idx, (data1, data2, data3) in enumerate(test_loader):
    data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)
    data1 = data1.cuda()
    data2 = data2.cuda()
    data3 = data3.cuda()

    # compute output
    # NOTE: reversing order of data because I'm not confident in changing down stream eval
    dista, distb, _, _, _ = tnet(data1, data3, data2)
    target = torch.FloatTensor(dista.size()).fill_(1)
    target = Variable(target)
    target = target.cuda()

    test_loss =  criterion(dista, distb, target).data[0]

    # measure accuracy and record loss
    acc = accuracy(dista, distb)
    accs.update(acc, data1.size(0))
    losses.update(test_loss, data1.size(0))      

  print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
      losses.avg, 100. * accs.avg))
  plotter.plot('acc', 'test', epoch, accs.avg)
  plotter.plot('loss', 'test', epoch, losses.avg)
  return accs.avg


def accuracy(dista, distb):
  margin = 0
  pred = (dista - distb - margin).cpu().data
  return (pred > 0).sum()*1.0/dista.size()[0]

def save_checkpoint(state, is_best, filename):
  # Saves checkpoint to disk
  directory = 'model_checkpoints/current/'
  if not os.path.exists(directory):
    os.makedirs(directory)
  filename = directory + filename
  torch.save(state, filename)
  torch.save(state, directory + 'most_recent.pth.tar')
  if is_best:
    shutil.copyfile(filename, directory + 'model_best.pth.tar')


def main():
  global plotter 
  plotter = VisdomLinePlotter(env_name="plooter")

  train_loader, test_loader = cptn.CurateTrainTest()

  model = Net()
  model = model.cuda()
  tnet = TripletNet(model)
  tnet = tnet.cuda()

  start_epoch = 1
  resume = False
  checkpoint_to_load = 'model_checkpoints/current/most_recent.pth.tar'
  best_acc = 0.0

  # optionally resume from a checkpoint
  if resume:
    if os.path.isfile(checkpoint_to_load):
      print("=> loading checkpoint '{}'".format(checkpoint_to_load))
      checkpoint = torch.load(checkpoint_to_load)
      start_epoch = checkpoint['epoch']
      best_acc = checkpoint['best_acc']
      tnet.load_state_dict(checkpoint['state_dict'])
      print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_to_load, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(checkpoint_to_load))

  margin = 1
  learning_rate = 0.001
  momentum = 0.9
  epochs = 20

  criterion = torch.nn.MarginRankingLoss(margin=margin)
  optimizer = optim.SGD(tnet.parameters(), lr=learning_rate, momentum=momentum)

  n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
  print('  + Number of params: {}'.format(n_parameters))

  for epoch in range(start_epoch, epochs + 1):
    # train for one epoch
    train(train_loader, tnet, criterion, optimizer, epoch)
    # evaluate on validation set
    acc = test(test_loader, tnet, criterion, epoch)

    print('current acc: ')
    print(acc)
    print('best acc: ')
    print(best_acc)

    # remember best acc and save checkpoint
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)

    save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': tnet.state_dict(),
      'best_acc': best_acc,
    }, is_best, 'checkpoint'+str(epoch)+'.pth.tar')


if __name__ == '__main__':
  main()
  print ("donzo")


