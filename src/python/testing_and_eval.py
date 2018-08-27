#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:35:43 2018

@author: samer

"""

from __future__ import print_function
import os
import torch
import shutil
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import convert_png_to_numpy as cptn

from sklearn import manifold
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets, transforms
from sklearn.decomposition import PCA, KernelPCA
from training_and_def import Net1, Net2, Net3, TripletNet, AverageMeter

#TODO: save database of past embeddings so don't have to run them through
#      network everytime

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]


def generateEmbeddings(test_set, model, embeddingD):
    model = model.cuda()
    model.eval()
    test_set_len = len(test_set)
    X = np.empty([test_set_len, embeddingD])
    for idx in range(test_set_len):
      if idx % 100 == 0:
        print(idx)

      datax_norm, datax_rot = test_set.getSpecificItem(idx)
      datax_rot = Variable(datax_rot, volatile=True)
      datax_norm = Variable(datax_norm, volatile=True)
      datax_rot = datax_rot.cuda()
      datax_norm = datax_norm.cuda()
      #NOTE: change depending on model
      #x = model(datax_norm, datax_rot)
      #x = model(datax_norm)
      x, r = model(datax_norm, datax_rot)

      x = x.cpu()
      npx = x.data.numpy()
      X[idx, :] = npx

    return X

def vizEmbeddingTrajectory(test_set, model, embeddingD):
    
    X = generateEmbeddings(test_set, model, embeddingD)

    print("Reducing dimensions...")
    #NOTE: Linear PCA
    #pca = PCA(n_components=3)
    #X_viz = pca.fit_transform(X)
    #NOTE: Kernel PCA
    #kpca = KernelPCA(kernel="rbf", n_components=3)
    #X_viz = kpca.fit_transform(X)
    #NOTE: Isomap
    X_viz = manifold.Isomap(n_neighbors=10, n_components=3).fit_transform(X)

    print("Creating vizualization")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = []
    ys = []
    zs = []
    test_set_len = len(test_set)
    colors = plt.cm.cool(np.linspace(0, 1, test_set_len))
    print(colors)
    for row in range(test_set_len):
      xs.append(X_viz[row, 0])
      ys.append(X_viz[row, 1])
      zs.append(X_viz[row, 2])

    ax.scatter(xs, ys, zs, c=colors, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def testHP(test_set, model):

    # switch to evaluation mode
    model.eval()
    confucius_matrix = []
    for idx in range(len(test_set)):
      one_row = []
      datax = test_set.getSpecificItem(idx)
      datax = Variable(datax, volatile=True)
      x = model(datax)
      for idy in range(len(test_set)):
        datay = test_set.getSpecificItem(idy)
        datay = Variable(datay)
        y = model(datay)

        z = x.sub(y)
        inv_sim = z.norm()
        one_row.append(inv_sim.data[0])

      confucius_matrix.append(one_row)

    print(confucius_matrix)


def testQuery(model, test_set, query_set):
    neighborhood_size = 10
    address_book = {}
    model.eval()
    for idx in range(len(query_set)):
      if idx % 1000 == 0:
        print(idx)

      datax = query_set.getSpecificItem(idx)
      datax = Variable(datax, volatile=True)
      x = model(datax)
      address_book[query_set.getAddress(idx)] = x

    query_results = []
    for idx in range(1, 3002, 300): # or the whole test set
      print(idx)
      datax = test_set.getSpecificItem(idx)
      datax = Variable(datax, volatile=True)
      x = model(datax)
      query_result = []
      for idt in range(len(query_set)):
        if idt % 1000 == 0:
          print(idt)
        fname = query_set.getAddress(idt)
        z = x.sub(address_book[fname])
        sim = z.norm()
        similarity = sim.data[0] #actually something like inverse similarity
        single_scan = []
        if len(query_result) < neighborhood_size:
          single_scan.append(fname)
          single_scan.append(address_book[fname])
          single_scan.append(similarity)
          query_result.append(single_scan)
        else:
          if similarity < query_result[0][2]:
            single_scan.append(fname)
            single_scan.append(address_book[fname])
            single_scan.append(similarity)
            query_result[0] = single_scan
            # reverse sort based on similarity
            query_result.sort(key = lambda k: (k[2]), reverse=True)

      query_results.append(query_result)

    #print(query_results)
    #for i in range(len(query_results)):
    #  print("QUERY: ")
    #  print(i)
    #  for j in range(len(query_results[i])):
    #    print(query_results[i][j][0]) #addresses of nearest neighbors
    #    print(query_results[i][j][2]) #similarities of nearest neighbors

    return query_results


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
    
    #print('acc: ')
    #print(acc)
    
    return accs.avg

def KNNOnRaw(database, queries):
    K = 10

    query_results = []
    #for idx in range(len(queries)): # or the whole test set
    for idx in range(1, 3002, 300): # or the whole test set
      print(idx)
      x = queries[idx]
      query_result = []
      for idt in range(len(database)):
        if idt % 1000 == 0:
          print(idt)
        y = database[idt]
        similarity = np.linalg.norm(y - x)
        single_scan = []
        if len(query_result) < K:
          single_scan.append(idt)
          single_scan.append(similarity)
          query_result.append(single_scan)
        else:
          if similarity < query_result[0][1]:
            single_scan.append(idt)
            single_scan.append(similarity)
            query_result[0] = single_scan
            # reverse sort based on similarity
            query_result.sort(key = lambda k: (k[1]), reverse=True)

      query_results.append(query_result)

    return query_results 

def vizRawDataTrajectory(all_scans):
    embeddingD = 1081
    test_set_len = len(all_scans)
    X = np.empty([test_set_len, embeddingD])
    for idx in range(test_set_len):
      if idx % 100 == 0:
        print(idx)
      X[idx, :] = all_scans[idx]
    
    print("Reducing dimensions...")
    #NOTE: Linear PCA
    #pca = PCA(n_components=3)
    #X_viz = pca.fit_transform(X)
    #NOTE: Kernel PCA
    #kpca = KernelPCA(kernel="rbf", n_components=3)
    #X_viz = kpca.fit_transform(X)
    #NOTE: Isomap
    X_viz = manifold.Isomap(n_neighbors=10, n_components=3).fit_transform(X)

    print("Creating vizualization")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = []
    ys = []
    zs = []
    colors = plt.cm.cool(np.linspace(0, 1, test_set_len))
    print(colors)
    for row in range(test_set_len):
      xs.append(X_viz[row, 0])
      ys.append(X_viz[row, 1])
      zs.append(X_viz[row, 2])

    ax.scatter(xs, ys, zs, c=colors, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def computeDimWiseVariance(all_vectors):
  dim_variances = []
  samples = []
  for i in range(len(all_vectors[0])):
    new_dim = []
    new_dim.append(all_vectors[0][i])
    samples.append(new_dim)

  for i in range(1, len(all_vectors)):
    for j in range(len(all_vectors[i])):
      samples[j].append(all_vectors[i][j])

  for i in range(len(samples)):
    mean_total = 0
    for j in range(len(samples[i])):
      mean_total = mean_total + samples[i][j]
    mean = mean_total / len(samples[i])

    var_total = 0
    for j in range(len(samples[i])):
      var_total = var_total + ((samples[i][j] - mean) * (samples[i][j] - mean))
    variance = var_total / (len(samples[i]) - 1)

    dim_variances.append(variance)

  return dim_variances

def main():
    #NOTE: KNN on Raw Data
    #database = cptn.LoadScansFromText("RawScansTRAINDB.txt")
    #queries = cptn.LoadScansFromText("RawScansTESTQUERY.txt")
    #query_results = KNNOnRaw(database, queries)
    #for i in range(len(query_results)):
    #  print("QUERY: "+str(i))
    #  for j in range(len(query_results[i])):
    #    print(query_results[i][j])

    #NOTE: ISOMAP on Raw Data
    #queries = cptn.LoadScansFromText("RawScansTESTQUERY.txt")
    #vizRawDataTrajectory(queries)

    #NOTE: Computing variance of Raw Data for Static Scenes
    #queries = cptn.LoadScansFromText("RawScansSS1.txt")
    #queries = cptn.LoadScansFromText("RawScansSS2.txt")
    #queries = cptn.LoadScansFromText("RawScansSS3.txt")
    #queries = cptn.LoadScansFromText("RawScansSS4.txt")
    #queries = cptn.LoadScansFromText("RawScansSS5.txt")
    #dim_vars = computeDimWiseVariance(queries)
    #plt.hist([dim_vars])
    #plt.hist([dim_vars], bins=[0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24])
    #plt.hist([dim_vars], bins=[0.0, 0.000625, 0.00125, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64])
    #NOTE: 1, 2, 3, 4, 5
    #plt.hist([dim_vars], bins=[0.0, 0.0000025, 0.000005,  0.00001, 0.00002, 0.00004, 0.00008, 0.00016, 0.00032, 0.00064])
    #plt.show()

    #NOTE: SELECT MODEL HERE #####
    #model = Net1()
    #checkpoint_to_load = 'model_checkpoints/archive/Inception_LeNet_Large/model_best.pth.tar'
    #checkpoint_to_load = 'model_checkpoints/archive/Inception_LeNet_Small/model_best.pth.tar'
    #model = Net2()
    #checkpoint_to_load = 'model_checkpoints/archive/Inception_FC_Large/model_best.pth.tar'
    #checkpoint_to_load = 'model_checkpoints/archive/Inception_FC_Small/model_best.pth.tar'
    model = Net3()
    #checkpoint_to_load = 'model_checkpoints/archive/Inception_LeNet_Decoder_Large/model_best.pth.tar'
    checkpoint_to_load = 'model_checkpoints/archive/Inception_LeNet_Decoder_Small/model_best.pth.tar'
    tnet = TripletNet(model)
    if os.path.isfile(checkpoint_to_load):
        print("=> loading checkpoint '{}'".format(checkpoint_to_load))
        checkpoint = torch.load(checkpoint_to_load)
        tnet.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_to_load, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_to_load))
        return
    ##############################

    #NOTE: Computing variance of Embeddings for Static Scenes
    #test_set = cptn.SpecialTestSet()
    #embeddingDim = 64
    #all_embeddings = generateEmbeddings(test_set, model, embeddingDim)
    #dim_vars = computeDimWiseVariance(all_embeddings)
    #plt.hist([dim_vars])
    #plt.show()

    #NOTE: ISOMAP on Embeddings
    test_set = cptn.SpecialTestSet()
    #embeddingDim = 64
    embeddingDim = 32
    vizEmbeddingTrajectory(test_set, model, embeddingDim)






    #train_loader, test_loader = cptn.CurateTrainTest()
    #test_set = cptn.SpecialTestSet()
    #query_set = cptn.SpecialQuerySet()

    #model = Net1()
    #model = Net2()
    #model = Net3()
    #tnet = TripletNet(model)

    #checkpoint_to_load = 'model_checkpoints/current/model_best.pth.tar'
    #if os.path.isfile(checkpoint_to_load):
    #    print("=> loading checkpoint '{}'".format(checkpoint_to_load))
    #    checkpoint = torch.load(checkpoint_to_load)
    #    tnet.load_state_dict(checkpoint['state_dict'])
    #    print("=> loaded checkpoint '{}' (epoch {})"
    #            .format(checkpoint_to_load, checkpoint['epoch']))
    #else:
    #    print("=> no checkpoint found at '{}'".format(checkpoint_to_load))
    #    return

    #margin = 1
    #criterion = torch.nn.MarginRankingLoss(margin=margin)

    #n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    #print('  + Number of params: {}'.format(n_parameters))

    #NOTE: for testing accuracy of given network on many different test sets
    #acc = test(test_loader, tnet, criterion)
    
    #NOTE: for testing intra test set similarity / embedding structure
    #testHP(test_set, model)

    #NOTE: for visualizing scan trajectories in embedding space
    #vizEmbeddingTrajectory(test_set, model)
    
    #NOTE: for testing similarity of query to result
    #query_results = testQuery(model, test_set, query_set)



if __name__ == '__main__':
    main()    
    print ("donzo")


