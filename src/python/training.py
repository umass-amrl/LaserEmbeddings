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

from network_definitions import VAE, TripletNet

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

#class WeightedPairwiseDistance(nn.Module):
#    def __init__(self):
#        super(WeightedPairwiseDistance, self).__init__()
#    def forward(self, rec, target, weights):
#        return weighted_pairwise_distance(rec, target, weights)

def weighted_pairwise_distance(rec, target, weights):

  #TODO: write unit test for this

  #test_in = np.asarray([[1, 1, 1],[-1, -1, -1]])
  #test_target = np.asarray([[0, 0, 1],[0, 1, 0]])
  #test_weights = np.asarray([[2, 2, 2],[1, 1, 1]])
  #test_in = torch.from_numpy(test_in)
  #test_target = torch.from_numpy(test_target)
  #test_weights = torch.from_numpy(test_weights)
  #test_in = Variable(test_in)
  #test_target = Variable(test_target)
  #test_weights = Variable(test_weights)
  #print("ORIGINAL")
  #print(test_weights)
  #test_weights = test_weights * -1.0
  #print("NEGATED")
  #print(test_weights)
  #test_in = test_in.cuda()
  #test_target = test_target.cuda()
  #test_weights = test_weights.cuda()

  #test_diff = torch.abs(test_in.sub(test_target))
  #print(test_diff.cpu().data.numpy())

  #test_diff = torch.sum(torch.mul(test_diff, test_weights))
  #print(test_diff.cpu().data.numpy())

  #print("RECREATION")
  #print(rec.cpu().data.numpy())
  #print("TARGET")
  #print(target.cpu().data.numpy())
  #print("WEIGHTS")
  #print(weights.cpu().data.numpy())
  rec = rec.cuda()
  target = target.cuda()
#  print(rec.is_cuda)
#  print(target.is_cuda)


  x = torch.abs(torch.add(rec, -target))
  return torch.sum(torch.mul(x, weights))
  #return torch.sum(torch.mul(torch.abs(torch.add(rec, -target)), weights))


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, weights):
    #BCE = F.binary_cross_entropy(recon_x.view(-1, 256 * 256), x.view(-1, 256 * 256), size_average=False)
    BCE = weighted_pairwise_distance(recon_x.view(-1, 256 * 256), x.view(-1, 256 * 256), weights)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, KLD



######################### TRAIN #########################

def train(train_loader, tnet, criterion1, criterion2, optimizer, epoch, margin):
  losses = AverageMeter()
  accs = AverageMeter()
  emb_norms = AverageMeter()

  # switch to train mode
  tnet.train()
  for batch_idx, (data1n, data2n, data3n, data1r, data2r, data3r, label) in enumerate(train_loader):
    #scan_x = data1n[:, 0, 0,:]
    rand_row = random.randint(0, 5)
    #rand_row = random.randint(0, 50)
    #rand_row = random.randint(0, 255)
    #scan_x = data1r[:, 0, rand_row,:]
    #scan_y = data2n[:, 0, 0,:]
    #scan_z = data3n[:, 0, 0,:]


#    scan_x = copy.deepcopy(data1n)
#    scan_y = copy.deepcopy(data2n)
#    scan_z = copy.deepcopy(data3n)

    #mask = np.zeros([256, 256])
    #scan_mask_x = scan_x[scan_x > 0.0]
    #print(scan_x.shape)
    #print("pre mask")
    #print(scan_x[0, :, :, :])
    #print("D1N pre")
    #print(data1n[0, :, :, :])
    #print(data1n.data.numpy()[0, :, :, :])
    batch_size = list(data1n.shape)[0]
    
    mask_x = copy.deepcopy(data1n)
    mask_x = np.reshape(mask_x, (batch_size, 256 * 256))
    #print(scan_mask_x.shape)
   

    #print("after reshape")
    #print(scan_x.data.numpy()[0, :, :, :])
    #print("D1N")
    #print(data1n.data.numpy()[0, :, :, :])

    total_pixels = float(batch_size * 256 * 256)
    pos_weight = total_pixels / float((0 < mask_x).sum())
    neg_weight = total_pixels / float(total_pixels - (0 < mask_x).sum())
    #print(pos_weight)
    #print(neg_weight)
    
    #NOTE: learns same recreation for everything, learns to ignore rear qudrant
    #scan_mask_x[scan_mask_x > 0] = pos_weight
    #scan_mask_x[scan_mask_x < 0] = neg_weight
    
    #NOTE: best so far --- VAE1
    #mask_x[mask_x > 0] = 100.0
    #mask_x[mask_x < 0] = 0.1
    
    #NOTE: 
    #scan_mask_x[scan_mask_x > 0] = 80.0
    #scan_mask_x[scan_mask_x < 0] = 1.0
    
    #NOTE: VAE3
    #mask_x[mask_x > 0] = 50.0
    #mask_x[mask_x < 0] = 0.1
    
    #NOTE: 
    mask_x[mask_x > 0] = 50.0
    mask_x[mask_x < 0] = 1.0
    
    #NOTE: after 100 epochs, learns only to make black image
    #scan_mask_x[scan_mask_x > 0] = 10.0
    #scan_mask_x[scan_mask_x < 0] = 1.0
    
    mask_x = Variable(mask_x)
    mask_x = mask_x.cuda()

    #print(scan_x[0, :])
    #print(rand_row)

    #print("post mask")
    #print(scan_x.data.numpy()[0, :, :, :])
    #print("D1N")
    #print(data1n.data.numpy()[0, :, :, :])
    
    #ds_scan = data2n[:, 0, 0,:]
    #scan_x = Variable(scan_x)
    #scan_x = scan_x.cuda()
    #scan_y = Variable(scan_y)
    #scan_y = scan_y.cuda()
    #scan_z = Variable(scan_z)
    #scan_z = scan_z.cuda()
    data1n, data2n, data3n = Variable(data1n), Variable(data2n), Variable(data3n)
    data1r, data2r, data3r = Variable(data1r), Variable(data2r), Variable(data3r)
    d1n = data1n.cuda()
    d2n = data2n.cuda()
    d3n = data3n.cuda()
    d1r = data1r.cuda()
    d2r = data2r.cuda()
    d3r = data3r.cuda()
    label = Variable(label)
    label = label.cuda()

#    print("post VAR")
#    print(scan_x.cpu().data.numpy()[0, :, :, :])
#    print("D1N")
#    print(d1n.cpu().data.numpy()[0, :, :, :])

    #print(label.shape)
    #print(label.data)
    label = label.squeeze(1)

 #   e_n1x, e_p1x, e_0x, r_x, e_n1y, e_p1y, e_0y, r_y, e_n1z, e_p1z, e_0z, r_z, sim = tnet(d1n, d2n, d3n, d1r, d2r, d3r, rand_row)
    rec_x, mu, logvar, _, _, _, _, _, _, _, _, _, _ = tnet(d1n, d2n, d3n, d1r, d2r, d3r, rand_row)



    loss, loss_embed = loss_function(rec_x, data1n, mu, logvar, mask_x)

    #rec_x = rec_x.view(-1, 512*512)
    #rec_x = rec_x.view(-1, 256*256)
    #print(rec_x.shape)
    #scan_x = scan_x.view(-1, 512*512)
    #data1n = data1n.view(-1, 256*256)
    #print(scan_x.shape)

    #print(scan_x.cpu().data)
    #print(rec_x.cpu().data)



    #print(scan_x[0, :])
    #print(r_x[0, 0, 0, :])
    #print(scan_x[0, :].shape)
    #print(scan_x.shape)
    #print(r_x.shape)
    #print(r_x[:, 0, 0, :].shape)
    #print(r_x[0, 0, 0, :].shape)

#    r_x = r_x[:, 0, :]
#    r_y = r_y[:, 0, :]
#    r_z = r_z[:, 0, :]
    
    #print(r_x.shape)
    #r_x = r_x[:, 0, 0, :]
    #r_y = r_y[:, 0, 0, :]
    #r_z = r_z[:, 0, 0, :]

 #   loss_rec_x = criterion2(rec_x, scan_x, scan_mask_x)

    #loss_rec_x_tensor = criterion2(scan_x, rec_x)
    #print(loss_rec_x_tensor.shape)
    #loss_rec_x = torch.mean(loss_rec_x_tensor)
    #loss_rec_x = torch.mean(criterion2(scan_x, rec_x))

#    loss_rec_y = torch.mean(criterion2(scan_y, r_y))
#    loss_rec_z = torch.mean(criterion2(scan_z, r_z))

##    loss_rec_emb_x = torch.mean(criterion2(e_n1x, e_p1x))
##    loss_rec_emb_y = torch.mean(criterion2(e_n1y, e_p1y))
##    loss_rec_emb_z = torch.mean(criterion2(e_n1z, e_p1z))

    #loss_embedd = 0.0001 * abs((e_0x.norm(2) + e_0y.norm(2) + e_0z.norm(2)) - 3.0)
    #loss_embedd = abs((e_0x.norm(2) + e_0y.norm(2) + e_0z.norm(2)) - 3.0)
#    loss_embedd = e_0x.norm(2) + e_0y.norm(2) + e_0z.norm(2)
    

    #loss_embedd = emb_x.norm(2)
    #loss_embedd = 1.0


    #loss_sim = 100.0 * criterion1(sim, label)

    #print(loss_sim.item())

    #loss = loss_rec_x+loss_rec_y+loss_rec_z+ loss_rec_emb_x+loss_rec_emb_y+loss_rec_emb_z + loss_sim+loss_embedd
    #loss = loss_rec_emb_x + loss_rec_emb_y + loss_rec_emb_z
    #loss = loss_rec_x + loss_rec_y + loss_rec_z + loss_embedd
    #loss = loss_rec_x + loss_rec_y + loss_rec_z + 0.0001 * loss_embedd
    #loss = loss_rec_x + 0.0001 * loss_embedd
    #loss = loss_rec_x
    #loss = loss_rec_x + loss_rec_y + loss_rec_z + loss_sim + loss_embedd
    #loss = loss_sim

    """
    emb_1, emb_2, emb_3, reconstruction_1 = tnet(data1n, data2n, data3n, data1r, data2r, data3r)

    #dist_a = F.pairwise_distance(embedded_1, embedded_2, 2) # L-2 norm
    #dist_b = F.pairwise_distance(embedded_1, embedded_3, 2) # L-2 norm
    dist_a = F.cosine_similarity(emb_1, emb_2) # cosine distance
    dist_b = F.cosine_similarity(emb_1, emb_3) # cosine distance

    # -1 means, dist_a should be smaller than dist_b
    target1 = torch.FloatTensor(dist_a.size()).fill_(-1)
    target1 = Variable(target1)
    target1 = target1.cuda()

    #rawoutfile = open('scansasimages.txt', 'a')
    #rawscanimg = ds_scan[0, :].data.tolist()
    #for i in range(len(rawscanimg)):
    #  rawoutfile.write(str(rawscanimg[i]))
    #  rawoutfile.write(" ")
    #rawoutfile.write("\n")
    #rawoutfile.close()

    #loss_triplet = criterion1(dist_a, dist_b, target1)
   
    loss_reconstruction = torch.mean(criterion2(ds_scan, reconstruction_1))

    #loss_embedd = emb_1.norm(2) + emb_2.norm(2) + emb_3.norm(2)
    #loss = 50.0 * loss_triplet + loss_reconstruction + 0.001 * loss_embedd
    #loss = loss_reconstruction + 0.001 * loss_embedd

    #print(50.0 * loss_triplet)
    #print(loss_reconstruction)
    #print(0.01 * loss_embedd)
    #print(loss)



    dh = emb_1 * (1 - emb_1) # Hadamard product produces size N_batch x N_hidden
    #W = tnet.embeddingnet.state_dict()['fc4.weight']
    #W = tnet.embeddingnet.state_dict()['fc3.weight']
    W = tnet.embeddingnet.state_dict()['fc2.weight']
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(Variable(W)**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    loss_contractive = torch.sum(torch.mm(dh**2, w_sum), 0)
    #loss_contractive = contractive_loss.mul_(lam)
    #loss = loss_reconstruction + 0.001 * loss_contractive
    loss = loss_reconstruction
    """

    # measure accuracy and record loss
    #acc = accuracy(dist_a, dist_b, margin)
    acc = 0.0
    losses.update(loss.item(), d1n.size(0))
    accs.update(acc, d1n.size(0))
    
    emb_norms.update(loss_embed.item(), d1n.size(0))
    #emb_norms.update(loss_embedd.item()/3.0, d1n.size(0))
    #emb_norms.update(loss_contractive.item()/3, data1n.size(0))


    # compute gradient and do optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    log_interval = 20
    #log_interval = 160
    #log_interval = 4
    #log_interval = 1

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{}]\t'
            'Loss: {:.4f} ({:.4f}) \t'
            'Acc: {:.2f}% ({:.2f}%) \t'
            'Emb_Norm: {:.2f} ({:.2f})'.format(
          epoch, batch_idx * len(data1n), len(train_loader.dataset),
          losses.val, losses.avg, 
          100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
  # log avg values to somewhere
##  plotter.plot('acc', 'train', epoch, accs.avg)
##  plotter.plot('loss', 'train', epoch, losses.avg)
##  plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)

  return losses.avg

######################### TEST #########################

def test(test_loader, tnet, criterion1, criterion2, epoch, margin):
  losses = AverageMeter()
  accs = AverageMeter()

  # switch to evaluation mode
  tnet.eval()
  for batch_idx, (data1n, data2n, data3n, data1r, data2r, data3r, label) in enumerate(test_loader):
    scan_x = data1n[:, 0, 0,:]
    scan_y = data2n[:, 0, 0,:]
    scan_z = data3n[:, 0, 0,:]
    #ds_scan = data2n[:, 0, 0,:]
    scan_x = Variable(scan_x)
    scan_x = scan_x.cuda()
    scan_y = Variable(scan_y)
    scan_y = scan_y.cuda()
    scan_z = Variable(scan_z)
    scan_z = scan_z.cuda()
    data1n, data2n, data3n = Variable(data1n), Variable(data2n), Variable(data3n)
    data1r, data2r, data3r = Variable(data1r), Variable(data2r), Variable(data3r)
    d1n = data1n.cuda()
    d2n = data2n.cuda()
    d3n = data3n.cuda()
    d1r = data1r.cuda()
    d2r = data2r.cuda()
    d3r = data3r.cuda()




    label = Variable(label)
    label = label.cuda()
    #print(label.shape)
    #print(label.data)
    label = label.squeeze(1)

    e_n1x, e_p1x, e_0x, r_x, e_n1y, e_p1y, e_0y, r_y, e_n1z, e_p1z, e_0z, r_z, sim = tnet(d1n, d2n, d3n, d1r, d2r, d3r, 0)

#    r_x = r_x[:, 0, :]
#    r_y = r_y[:, 0, :]
#    r_z = r_z[:, 0, :]
    
    #r_x = r_x[:, 0, 0, :]
    #r_y = r_y[:, 0, 0, :]
    #r_z = r_z[:, 0, 0, :]
    
    loss_rec_x = torch.mean(criterion2(scan_x, r_x))
    loss_rec_y = torch.mean(criterion2(scan_y, r_y))
    loss_rec_z = torch.mean(criterion2(scan_z, r_z))

 #   loss_rec_emb_x = torch.mean(criterion2(e_n1x, e_p1x))
 #   loss_rec_emb_y = torch.mean(criterion2(e_n1y, e_p1y))
 #   loss_rec_emb_z = torch.mean(criterion2(e_n1z, e_p1z))

    loss_embedd = 0.0001 * (e_0x.norm(2) + e_0y.norm(2) + e_0z.norm(2))
    #loss_sim = 100.0 * criterion1(sim, label)

    #print(loss_sim.item())

    #loss = loss_rec_x+loss_rec_y+loss_rec_z+loss_rec_emb_x+loss_rec_emb_y+loss_rec_emb_z+loss_sim + loss_embedd
    #loss = loss_rec_emb_x + loss_rec_emb_y + loss_rec_emb_z
    #test_loss = loss_rec_x + loss_rec_y + loss_rec_z
    test_loss = loss_rec_x
    #loss = loss_rec_x + loss_rec_y + loss_rec_z + loss_sim + loss_embedd
    #loss = loss_sim






















#    emb_1, emb_2, emb_3, reconstruction_1 = tnet(data1n, data2n, data3n, data1r, data2r, data3r)


    #dist_a = F.pairwise_distance(embedded_x, embedded_y, 2) # L-2 norm
    #dist_b = F.pairwise_distance(embedded_x, embedded_z, 2) # L-2 norm
#    dist_a = F.cosine_similarity(emb_1, emb_2) # cosine distance
#    dist_b = F.cosine_similarity(emb_1, emb_3) # cosine distance


#    target = torch.FloatTensor(dist_a.size()).fill_(-1)
#    target = Variable(target)
#    target = target.cuda()

    #test_loss = criterion(dist_a, dist_b, target).data[0]
#    test_loss = torch.mean(criterion2(ds_scan, reconstruction_1))

    # measure accuracy and record loss
#    acc = accuracy(dist_a, dist_b, margin)
    acc = 0.0
    accs.update(acc, data1n.size(0))
    losses.update(test_loss, data1n.size(0))

  print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
      losses.avg, 100. * accs.avg))
##  plotter.plot('acc', 'test', epoch, accs.avg)
##  plotter.plot('loss', 'test', epoch, losses.avg)
  return accs.avg




def accuracy(dista, distb, margin):
  pred = (distb - dista - margin).cpu().data
  return float((pred > 0).sum())/float(dista.size()[0])

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
  #plotter = VisdomLinePlotter(env_name="plooter")
  plotter = VisdomLinePlotter()

  laser_dataset_train, laser_dataset_test = cptn.CurateTrainTest()
  #train_loader, laser_dataset_test = cptn.CurateTrainTest()
  #train_loader, test_loader = cptn.CrossedOverSets()
  train_loader = torch.utils.data.DataLoader(laser_dataset_train, batch_size=16, shuffle=True)
  #train_loader = torch.utils.data.DataLoader(laser_dataset_train, batch_size=256, shuffle=True)
  test_loader = torch.utils.data.DataLoader(laser_dataset_test, batch_size=4, shuffle=False)
  #test_loader = torch.utils.data.DataLoader(laser_dataset_test, batch_size=64, shuffle=False)

  #corrupt_train_loader = cptn.CorruptedSets(laser_dataset_train)
  #objectified_train_loader = cptn.ObjectifiedSets(laser_dataset_train)

  model = VAE()
  #model = ConvNet()
  #model = StackedAutoEncoder()
  #model = Net3()
  #model = Netnet()
  model = model.cuda()
  tnet = TripletNet(model)
  tnet = tnet.cuda()

  start_epoch = 1
  resume = True
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
  #margin1a = 0.4
  margin1a = 0.5
  margin1b = 0.01
  #learning_rate = 0.001
  #init_learning_rate = 0.1
  #init_learning_rate = 0.005
  #init_learning_rate = 0.003
  init_learning_rate = 0.001
  #init_learning_rate = 0.002
  #init_learning_rate = 0.0005
  #momentum = 0.3
  momentum = 0.9
  #momentum = 0.1
  #epochs = 20
  #epochs = 3
  #epochs = 400
  epochs = 4000

  #NOTE: For net 3
  #criterion1a = torch.nn.MarginRankingLoss(margin=margin1a)
  #criterion1b = torch.nn.MarginRankingLoss(margin=margin1b)
  criterion1 = torch.nn.CrossEntropyLoss()
  criterion2 = torch.nn.CrossEntropyLoss()
  #criterion1 = torch.nn.CosineEmbeddingLoss(margin=margin)
  #criterion2 = torch.nn.PairwiseDistance(p=2)
  #criterion2 = torch.nn.PairwiseDistance(p=1)
  #criterion2 = WeightedPairwiseDistance()

  n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
  print('  + Number of params: {}'.format(n_parameters))

  #best_avg_loss = 512.0
  best_avg_loss = 5000000.0

  learning_rate = init_learning_rate
  optimizer = optim.Adam(tnet.parameters())
  #optimizer = optim.SGD(tnet.parameters(), lr=learning_rate, momentum=momentum)
  for epoch in range(start_epoch, epochs + 1):
    # train for one epoch
    #learning_rate = init_learning_rate * 0.99**epoch
    #learning_rate = init_learning_rate
    #print("learning rate: '{}'".format(learning_rate))
    #optimizer = optim.SGD(tnet.parameters(), lr=learning_rate, momentum=momentum)
    #avg_loss = train(train_loader, tnet, criterion1a, criterion2, optimizer, epoch, margin1a)
    avg_loss = train(train_loader, tnet, criterion1, criterion2, optimizer, epoch, margin1a)
    #train(corrupt_train_loader, tnet, criterion1a, criterion2, optimizer, epoch, margin1a)
    #train(objectified_train_loader, tnet, criterion1b, criterion2, optimizer, epoch, margin1b)


    is_least_lossy = avg_loss < best_avg_loss
    best_avg_loss = min(avg_loss, best_avg_loss)


    # evaluate on validation set
    #TODO: turn on again
    #acc = test(test_loader, tnet, criterion1, criterion2, epoch, margin1a)
    acc = 0.0
    print('current acc: ')
    print(acc)
    print('best acc: ')
    print(best_acc)

    # remember best acc and save checkpoint
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)


    if is_least_lossy:
      save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': tnet.state_dict(),
        'best_acc': best_acc,
      }, is_best, 'checkpoint'+str(epoch)+'.pth.tar')


if __name__ == '__main__':
  print ("starting")
  main()
  print ("donzo")


