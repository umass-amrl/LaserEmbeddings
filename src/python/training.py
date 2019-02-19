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
import sys
import copy
import torch
import random
import shutil
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import convert_png_to_numpy as cptn
import torch.backends.cudnn as cudnn

from network_definitions import VAE, SimNet, SimNetTrainer

from visdom import Visdom
from torch.autograd import Variable
from torchvision import datasets, transforms

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

######################### LOSS FUNCTIONS #########################

def weighted_pairwise_distance(rec, target, weights):
  rec = rec.cuda()
  target = target.cuda()
  x = torch.abs(torch.add(rec, -target))
  return torch.sum(torch.mul(x, weights))

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, weights):
    BCE = weighted_pairwise_distance(recon_x.view(-1, 256 * 256), x.view(-1, 256 * 256), weights)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, KLD

######################### TRAIN #########################

def trainSimNet(train_loader, vae, simnet_trainer, epoch, log_interval):
  losses = AverageMeter()
  vae.eval()
  simnet_trainer.train()
  for batch_idx, (scan_A, scan_B, scan_Q, label) in enumerate(train_loader):
    #TODO: get the right label

    with torch.no_grad():
      scan_A = Variable(scan_A)
      scan_B = Variable(scan_B)
      scan_Q = Variable(scan_Q)
      scan_A = scan_A.cuda()
      scan_B = scan_B.cuda()
      scan_Q = scan_Q.cuda()
      _, mu_A, _ = vae(scan_A)
      _, mu_B, _ = vae(scan_B)
      _, mu_Q, _ = vae(scan_Q)

    sim_score_A, sim_score_B = simnet_trainer(mu_A, mu_B, mu_Q)

    #TODO: compute loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #TODO: may need a different way of getting batch size
    losses.update(loss.item(), scan_Q.size(0))
    emb_norms.update(loss_embed.item(), scan_Q.size(0))
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{}]\t'
            'Loss: {:.4f} ({:.4f}) \t'
            'Emb_Norm: {:.2f} ({:.2f})'.format(
          epoch, batch_idx * len(scan_Q), len(train_loader.dataset),
          losses.val, losses.avg, emb_norms.val, emb_norms.avg))

  return losses.avg

def trainVAE(train_loader, vae, optimizer, epoch, log_interval):
  losses = AverageMeter()
  emb_norms = AverageMeter()

  # switch to train mode
  vae.train()
  for batch_idx, (input_scan, recreation_target) in enumerate(train_loader):

    batch_size = list(input_scan.shape)[0]
    mask_x = copy.deepcopy(recreation_target)
    mask_x = np.reshape(mask_x, (batch_size, 256 * 256))
    total_pixels = float(batch_size * 256 * 256)
    #pos_weight = total_pixels / float((0 < mask_x).sum())
    #neg_weight = total_pixels / float(total_pixels - (0 < mask_x).sum())
    pos_weight = 50.0
    neg_weight = 1.0
    mask_x[mask_x > 0] = pos_weight
    mask_x[mask_x < 0] = neg_weight

    input_scan = Variable(input_scan)
    input_scan = input_scan.cuda()
    recreation_target = Variable(recreation_target)
    recreation_target = recreation_target.cuda()
    mask_x = Variable(mask_x)
    mask_x = mask_x.cuda()

    rec_x, mu, logvar = vae(input_scan)

    loss, loss_embed = loss_function(rec_x, recreation_target, mu, logvar, mask_x)

    # compute gradient and do optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), input_scan.size(0))
    emb_norms.update(loss_embed.item(), input_scan.size(0))
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{}]\t'
            'Loss: {:.4f} ({:.4f}) \t'
            'Emb_Norm: {:.2f} ({:.2f})'.format(
          epoch, batch_idx * len(input_scan), len(train_loader.dataset),
          losses.val, losses.avg, emb_norms.val, emb_norms.avg))
  # log avg values to somewhere
##  plotter.plot('acc', 'train', epoch, accs.avg)
##  plotter.plot('loss', 'train', epoch, losses.avg)
##  plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)

  return losses.avg

######################### TEST #########################

def testSimNet(test_loader, vae, simnet_trainer, epoch):
  accs = AverageMeter()
  vae.eval()
  simnet.eval()
  with torch.no_grad():
    for batch_idx, (scan_A, scan_B, scan_Q) in enumerate(test_loader):
      #TODO: need labels from dataset

      scan_A = Variable(scan_A)
      scan_B = Variable(scan_B)
      scan_Q = Variable(scan_Q)
      scan_A = scan_A.cuda()
      scan_B = scan_B.cuda()
      scan_Q = scan_Q.cuda()
      _, mu_A, _ = vae(scan_A)
      _, mu_B, _ = vae(scan_B)
      _, mu_Q, _ = vae(scan_Q)
      sim_score_A, sim_score_B = simnet_trainer(mu_A, mu_B, mu_Q)



      #TODO: compute accuracy
      acc = 0.0
      accs.update(acc, scan_Q.size(0))

  return accs.avg

def testVAE(test_loader, vae, epoch):
  losses = AverageMeter()

  # switch to evaluation mode
  vae.eval()
  with torch.no_grad():
    for batch_idx, (input_scan, recreation_target) in enumerate(test_loader):

      batch_size = list(input_scan.shape)[0]
      mask_x = copy.deepcopy(input_scan)
      mask_x = np.reshape(mask_x, (batch_size, 256 * 256))
      total_pixels = float(batch_size * 256 * 256)
      #pos_weight = total_pixels / float((0 < mask_x).sum())
      #neg_weight = total_pixels / float(total_pixels - (0 < mask_x).sum())
      pos_weight = 50.0
      neg_weight = 1.0
      mask_x[mask_x > 0] = pos_weight
      mask_x[mask_x < 0] = neg_weight

      input_scan = Variable(input_scan)
      input_scan = input_scan.cuda()
      recreation_target = Variable(recreation_target)
      recreation_target = recreation_target.cuda()
      mask_x = Variable(mask_x)
      mask_x = mask_x.cuda()

      rec_x, mu, logvar = vae(input_scan)
      loss, loss_embed = loss_function(rec_x, recreation_target, mu, logvar, mask_x)

      losses.update(loss, input_scan.size(0))

  print('\nTest set: Average loss: {:.4f}\n'.format(losses.avg))
##  plotter.plot('acc', 'test', epoch, accs.avg)
##  plotter.plot('loss', 'test', epoch, losses.avg)
  return losses.avg


######################### SAVING AND LOADING #########################

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

def shouldSave(model, best_train, best_test, epoch):
  is_best = True
  best_acc = 0.0
  if best_train:
    save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc': best_acc,
    }, is_best, 'checkpoint_train_'+str(epoch)+'.pth.tar')

  if best_test:
    save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc': best_acc,
    }, is_best, 'checkpoint_test_'+str(epoch)+'.pth.tar')

def loadNetwork(model, checkpoint_to_load):
  model = model.cuda()

  epoch = 1
  best_acc = 0.0

  if os.path.isfile(checkpoint_to_load):
    print("=> loading checkpoint '{}'".format(checkpoint_to_load))
    checkpoint = torch.load(checkpoint_to_load)
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(checkpoint_to_load, checkpoint['epoch']))
  else:
    print("=> no checkpoint found at '{}'".format(checkpoint_to_load))

  n_parameters = sum([p.data.nelement() for p in model.parameters()])
  print('  + Number of params: {}'.format(n_parameters))

  return model, epoch, best_acc

############################## MAIN ##############################

def main():
  mode = -1
  if len(sys.argv) < 1:
    print("No mode selected. Usage: python training.py <mode>")
    return
  else:
    mode = int(sys.argv[1])

  if mode != 0 and mode != 1:
    print("Unrecognized mode. Exiting")
    return

  model = SimNet()
  checkpoint_to_load = ""

  if mode == 0: # Train Variational Autoencoder
    model = VAE()
    checkpoint_to_load = 'model_checkpoints/archive/Cartesian/VAE_repeat_init/checkpoint_train_15.pth.tar'

  elif mode == 1: # Train SimNet
    model = SimNetTrainer(model)
    #checkpoint_to_load = 'model_checkpoints/current/'

  #TODO: set up databases for different training techniques and different networks
  laser_dataset_train, laser_dataset_test = cptn.CurateTrainTest()
  train_loader = torch.utils.data.DataLoader(laser_dataset_train, batch_size=16, shuffle=True)
  test_loader = torch.utils.data.DataLoader(laser_dataset_test, batch_size=4, shuffle=False)

  start_epoch = 1
  resume = True
  best_acc = 0.0

  # optionally resume from a checkpoint
  if resume:
    model, start_epoch, best_acc = loadNetwork(model, checkpoint_to_load)
  model = model.cuda()

  #learning_rate = 0.001
  #momentum = 0.9
  #optimizer = optim.SGD(tnet.parameters(), lr=learning_rate, momentum=momentum)
  optimizer = optim.Adam(model.parameters())

  #TODO: sim ranking loss

  epochs = 4000
  log_interval = 20
  best_avg_loss_train = 5000000.0
  best_avg_loss_test = 5000000.0
  best_avg_acc_test = 0.0
  is_least_lossy_train = False
  is_least_lossy_test = False
  is_most_acc_test = False
  for epoch in range(start_epoch, epochs + 1):
    if mode == 0:
      avg_loss_train = trainVAE(train_loader, model, optimizer, epoch, log_interval)
      is_least_lossy_train = avg_loss_train < best_avg_loss_train
      best_avg_loss_train = min(avg_loss_train, best_avg_loss_train)

      avg_loss_test = testVAE(test_loader, model, epoch)
      is_least_lossy_test = avg_loss_test < best_avg_loss_test
      best_avg_loss_test = min(avg_loss_test, best_avg_loss_test)

      shouldSave(model, is_least_lossy_train, is_least_lossy_test, epoch)

    elif mode == 1:
      avg_loss_train = trainSimNet(train_loader, vae, model, optimizer, epoch, log_interval)
      is_least_lossy_train = avg_loss_train < best_avg_loss_train
      best_avg_loss_train = min(avg_loss_train, best_avg_loss_train)

      avg_acc_test = testSimNet(test_loader, vae, model, epoch)
      is_most_acc_test = avg_acc_test < best_avg_acc_test
      best_avg_acc_test = min(avg_acc_test, best_avg_acc_test)

      shouldSave(model, is_least_lossy_train, is_most_acc_test, epoch)

if __name__ == '__main__':
  main()
  print ("donzo")


