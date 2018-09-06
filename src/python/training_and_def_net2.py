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

class Net2(nn.Module): #NOTE: 1D conv (inception style) w/ straight to large FC
  def __init__(self):
    super(Net2, self).__init__()

    # non-square kernel 
    self.conv1_1 = nn.Conv2d(1, 1, (1, 3), 1, (0, 1))
    self.conv1_2 = nn.Conv2d(1, 1, (1, 5), 1, (0, 2))
    self.conv1_3 = nn.Conv2d(1, 1, (1, 9), 1, (0, 4))
    self.conv1_4 = nn.Conv2d(1, 1, (1, 17), 1, (0, 8))
    self.conv1_5 = nn.Conv2d(1, 1, (1, 33), 1, (0, 16))
    self.conv1_6 = nn.Conv2d(1, 1, (1, 65), 1, (0, 32))
    self.conv1_7 = nn.Conv2d(1, 1, (1, 129), 1, (0, 64))

    self.fc1 = nn.Linear(7 * 64 * 64, 256)
    self.fc2 = nn.Linear(256, 128)

    #NOTE: BIG embeddings
    #self.fc3 = nn.Linear(128, 64)

    #NOTE: small embeddings
    self.fc3 = nn.Linear(128, 32)

  def forward(self, xr):
    xr1 = F.relu(self.conv1_1(xr))
    xr2 = F.relu(self.conv1_2(xr))
    xr3 = F.relu(self.conv1_3(xr))
    xr4 = F.relu(self.conv1_4(xr))
    xr5 = F.relu(self.conv1_5(xr))
    xr6 = F.relu(self.conv1_6(xr))
    xr7 = F.relu(self.conv1_7(xr))
    
    xr1 = F.max_pool2d(xr1, (4, 4))
    xr2 = F.max_pool2d(xr2, (4, 4))
    xr3 = F.max_pool2d(xr3, (4, 4))
    xr4 = F.max_pool2d(xr4, (4, 4))
    xr5 = F.max_pool2d(xr5, (4, 4))
    xr6 = F.max_pool2d(xr6, (4, 4))
    xr7 = F.max_pool2d(xr7, (4, 4))

    outputs = [xr1, xr2, xr3, xr4, xr5, xr6, xr7]
    x = torch.cat(outputs, 1)

    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))

    #NOTE: BIG embeddings
    #x = self.fc3(x)

    #NOTE: small embeddings
    x = self.fc3(x)

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

  def forward(self, x, y, z):
    embedded_x = self.embeddingnet(x)
    embedded_y = self.embeddingnet(y)
    embedded_z = self.embeddingnet(z)
    #TODO: try different distances (cosine, etc.)
    dist_a = F.pairwise_distance(embedded_x, embedded_y, 2) # L-2 norm
    dist_b = F.pairwise_distance(embedded_x, embedded_z, 2) # L-2 norm
    return dist_a, dist_b, embedded_x, embedded_y, embedded_z

############################## STATISTICS ##############################

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

######################### TWO #########################

def train2(train_loader, tnet, criterion, optimizer, epoch):
  losses = AverageMeter()
  accs = AverageMeter()
  emb_norms = AverageMeter()

  # switch to train mode
  tnet.train()
  for batch_idx, (data1n, data2n, data3n, data1r, data2r, data3r) in enumerate(train_loader):
    data1r, data2r, data3r = Variable(data1r), Variable(data2r), Variable(data3r)
    data1r = data1r.cuda()
    data2r = data2r.cuda()
    data3r = data3r.cuda()

    # compute output
    # NOTE: reversing order of data because I'm not confident in changing down stream eval
    dista, distb, embedded_x, embedded_y, embedded_z = tnet(data1r, data3r, data2r)
    # 1 means, dista should be larger than distb
    target = torch.FloatTensor(dista.size()).fill_(1)
    target = Variable(target)
    target = target.cuda()

    loss_triplet = criterion(dista, distb, target)
    loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
    loss = loss_triplet + 0.001 * loss_embedd

    # measure accuracy and record loss
    acc = accuracy(dista, distb)
    losses.update(loss_triplet.data[0], data1r.size(0))
    accs.update(acc, data1r.size(0))
    emb_norms.update(loss_embedd.data[0]/3, data1r.size(0))

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
          epoch, batch_idx * len(data1r), len(train_loader.dataset),
          losses.val, losses.avg, 
          100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
  # log avg values to somewhere
  plotter.plot('acc', 'train', epoch, accs.avg)
  plotter.plot('loss', 'train', epoch, losses.avg)
  plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)


def test2(test_loader, tnet, criterion, epoch):
  losses = AverageMeter()
  accs = AverageMeter()

  # switch to evaluation mode
  tnet.eval()
  for batch_idx, (data1n, data2n, data3n, data1r, data2r, data3r) in enumerate(test_loader):
    data1r, data2r, data3r = Variable(data1r), Variable(data2r), Variable(data3r)
    data1r = data1r.cuda()
    data2r = data2r.cuda()
    data3r = data3r.cuda()

    # compute output
    # NOTE: reversing order of data because I'm not confident in changing down stream eval
    dista, distb, _, _, _ = tnet(data1r, data3r, data2r)
    target = torch.FloatTensor(dista.size()).fill_(1)
    target = Variable(target)
    target = target.cuda()

    test_loss =  criterion(dista, distb, target).data[0]

    # measure accuracy and record loss
    acc = accuracy(dista, distb)
    accs.update(acc, data1r.size(0))
    losses.update(test_loss, data1r.size(0))

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

############################## MAIN ##############################

def main():
  global plotter 
  plotter = VisdomLinePlotter(env_name="plooter")

  train_loader, test_loader = cptn.CurateTrainTest()

  model = Net2()
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

  #margin = 1
  margin = 0.4
  learning_rate = 0.001
  momentum = 0.9
  #epochs = 20
  #epochs = 3
  epochs = 10

  #criterion = torch.nn.MarginRankingLoss(margin=margin)
  #NOTE: For net 3
  #criterion1 = torch.nn.MarginRankingLoss(margin=margin)
  criterion1 = torch.nn.CosineEmbeddingLoss(margin=margin)
  criterion2 = torch.nn.PairwiseDistance(p=2)
  optimizer = optim.SGD(tnet.parameters(), lr=learning_rate, momentum=momentum)

  n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
  print('  + Number of params: {}'.format(n_parameters))

  for epoch in range(start_epoch, epochs + 1):
    # train for one epoch
    train2(train_loader, tnet, criterion, optimizer, epoch)
    
    # evaluate on validation set
    acc = test2(test_loader, tnet, criterion, epoch)

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


