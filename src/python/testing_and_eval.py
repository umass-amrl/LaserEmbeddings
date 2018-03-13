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
#import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from torchvision import datasets, transforms
#from visdom import Visdom
from training_and_def import Net, TripletNet, AverageMeter

#TODO: make todo list

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]


def main():
    test_loader = cptn.SpecialTestSet()
    
    model = Net()
    tnet = TripletNet(model)

    checkpoint_to_load = 'model_checkpoints/current/model_best.pth.tar'
    if os.path.isfile(checkpoint_to_load):
        print("=> loading checkpoint '{}'".format(checkpoint_to_load))
        checkpoint = torch.load(checkpoint_to_load)
        tnet.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_to_load, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_to_load))
        return

    margin = 1
    criterion = torch.nn.MarginRankingLoss(margin=margin)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    acc = test(test_loader, tnet, criterion)
    print('acc: ')
    print(acc)


def test(test_loader, tnet, criterion):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        #if args.cuda:
        #    data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        # NOTE: reversing order of data because I'm not confident in changing down stream eval
        dista, distb, _, _, _ = tnet(data1, data3, data2)
        target = torch.FloatTensor(dista.size()).fill_(1)
        #if args.cuda:
        #    target = target.cuda()
        target = Variable(target)
        test_loss =  criterion(dista, distb, target).data[0]

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))      

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    
    return accs.avg


if __name__ == '__main__':
    main()    
    print ("donzo")


