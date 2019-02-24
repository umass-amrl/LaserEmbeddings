#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:35:43 2018

@author: samer

"""

from __future__ import print_function
import os
import sys
import torch
import shutil
import rosbag
import argparse
import scipy.misc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import convert_png_to_numpy as cptn
from network_definitions import VAE, SimNet, SimNetTrainer
from sklearn import manifold
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets, transforms
from sklearn.decomposition import PCA, KernelPCA

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

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














############################## VISUALIZE TRAJECTORIES ##############################

def generateDataTrajectories(all_vectors):
  trajectories = []
  print("Reducing dimensions...")
  # Linear PCA
  pca = PCA(n_components=3)
  viz_pca = pca.fit_transform(all_vectors)
  trajectories.append(viz_pca)
  # Kernel PCA
  kpca = KernelPCA(kernel="rbf", n_components=3)
  viz_kpca = kpca.fit_transform(all_vectors)
  trajectories.append(viz_kpca)
  # Isomap
  viz_isomap = manifold.Isomap(n_neighbors=10, n_components=3).fit_transform(all_vectors)
  trajectories.append(viz_isomap)

  return trajectories

def vizDataTrajectories(trajectories):
  print("Creating vizualizations")

  for traj in trajectories:
    traj_len = traj[:, 0].size
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = []
    ys = []
    zs = []
    colors = plt.cm.cool(np.linspace(0, 1, traj_len))
    print(colors)
    for row in range(traj_len):
      xs.append(traj[row, 0])
      ys.append(traj[row, 1])
      zs.append(traj[row, 2])

    ax.scatter(xs, ys, zs, c=colors, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

  plt.show()

############################## COMPUTE VARIANCE ##############################

def computeDimWiseVariance(all_vectors):
  print(all_vectors.shape)
  dim_variances = np.empty([1, all_vectors[0, :].size])
  print(dim_variances.shape)
  
  for i in range(all_vectors[0, :].size):
    dim_variances[0, i] = all_vectors[:, i].var(0)

  return dim_variances

############################## WRITING STUFF ##############################

def writeRecreationsToText(recreations):
  outfile = open('recreations.txt', 'a')
  for i in range(recreations[:, 0].size):
    recreation = recreations[i, :]
    for j in range(recreation.size):
      outfile.write(str(recreation[j]))
      outfile.write(" ")
    outfile.write("\n")

  outfile.close();

def writeEmbeddingsToText(embeddings):
  outfile = open('embeddings.txt', 'a')
  for i in range(embeddings[:, 0].size):
    embedding = embedding[i, :]
    for j in range(embedding.size):
      outfile.write(str(embedding[j]))
      outfile.write(" ")
    outfile.write("\n")

  outfile.close();

def writeDownsampledInputs(test_set):
  outfile = open('medianfiltered.txt', 'a')
  for idx in range(len(test_set)):
    dn, dr = test_set.getSpecificItem(idx)
    dn = dn[0, 0, :].data.numpy()
    for j in range(dn.size):
      outfile.write(str(dn[j]))
      outfile.write(" ")
    outfile.write("\n")

  outfile.close();

############################## GENERATING STUFF ##############################

def generateEmbeddings(test_set, model):
  model.eval()
  test_set_len = len(test_set)
  input_scan, _ = test_set[0]
  input_scan = input_scan.unsqueeze(0)
  with torch.no_grad():
    input_scan = Variable(input_scan)
  input_scan = input_scan.cuda()
  _, emb, _ = model(input_scan)
  emb = emb.cpu()
  np_emb = emb.data.numpy()
  embeddings = np.empty([test_set_len, np_emb.size])
  for idx in range(test_set_len):
    if idx % 100 == 0:
      print(idx)

    input_scan, _ = test_set[idx]
    input_scan = input_scan.unsqueeze(0)
    with torch.no_grad():
      input_scan = Variable(input_scan)
    input_scan = input_scan.cuda()
    _, emb, _ = model(input_scan)

    emb = emb.cpu()
    np_emb = emb.data.numpy()
    embeddings[idx, :] = np_emb

  return embeddings

def generateInterpolatedEmbeddings(test_set, model):
  model = model.cuda()
  model.eval()
  test_set_len = len(test_set)
  real_scan_interval = 200
  interpolation_density = 10
  num_embeddings = (test_set_len / real_scan_interval) * interpolation_density + 1
  input_scan, _ = test_set[0]
  input_scan = input_scan.unsqueeze(0)
  with torch.no_grad():
    input_scan = Variable(input_scan)
  input_scan = input_scan.cuda()
  _, emb, _ = model(input_scan)
  emb = emb.cpu()
  np_emb = emb.data.numpy()
  interpolated_embeddings = np.empty([num_embeddings, np_emb.size])
  real_scans = []
  for idx in range(0, test_set_len, real_scan_interval):

    input_scan, _ = test_set[idx]
    input_scan = input_scan.unsqueeze(0)
    with torch.no_grad():
      input_scan = Variable(input_scan)
    input_scan = input_scan.cuda()
    _, emb, _ = model(input_scan)
    emb = emb.cpu()
    np_emb = emb.data.numpy()
    real_scans.append(np_emb)

  row = 0
  for idx in range(len(real_scans)-1):
    interpolated_embeddings[row, :] = real_scans[idx]
    row = row + 1
    for i in range(1, interpolation_density):
      embedding_diff = real_scans[idx + 1] - real_scans[idx]
      interpolated_embedding = real_scans[idx] + float(i)/float(interpolation_density) * embedding_diff
      interpolated_embeddings[row, :] = interpolated_embedding
      row = row + 1

  interpolated_embeddings[row, :] = real_scans[-1]


  return interpolated_embeddings

def generateScanRecreationsFromData(test_set, model):
  model = model.cuda()
  model.eval()
  test_set_len = len(test_set)

  input_scan, _ = test_set[0]
  input_scan = input_scan.unsqueeze(0)
  with torch.no_grad():
    input_scan = Variable(input_scan)
  input_scan = input_scan.cuda()
  rec, _, _ = model(input_scan)
  rec = rec.cpu()
  rec = rec.data.numpy()
  rec = rec.squeeze(0)
  rec = rec.squeeze(0)
  rec = np.reshape(rec, (1, 256 * 256))
  recreations = np.empty([test_set_len, rec.size])
  for idx in range(test_set_len):
    if idx % 100 == 0:
      print(idx)

    input_scan, _ = test_set[idx]
    input_scan = input_scan.unsqueeze(0)
    with torch.no_grad():
      input_scan = Variable(input_scan)
    input_scan = input_scan.cuda()

    rec, _, _ = model(input_scan)

    rec = rec.cpu()
    np_rec = rec.data.numpy()
    np_rec = np_rec.squeeze(0)
    np_rec = np_rec.squeeze(0)

    np_in = input_scan.cpu().data.numpy()
    np_in = np_in.squeeze(0)
    np_in = np_in.squeeze(0)

    np_rec[np_rec > 0.0] = 255
    np_rec[np_rec < 0.0] = 0

    if idx % 100 == 0:
      ofname = "sample_results/orig_"+str(idx+1)+".png"
      rfname = "sample_results/rec_"+str(idx+1)+".png"
      scipy.misc.imsave(rfname, np_rec)
      scipy.misc.imsave(ofname, np_in)

    np_rec = np.reshape(np_rec, (1, 256 * 256))
    recreations[idx, :] = np_rec

  return recreations

def generateScanRecreationsFromEmbeddings(embeddings, model):
  model = model.cuda()
  model.eval()

  recreations = np.empty([embeddings[:, 0].size, 256 * 256])
  for idx in range(len(embeddings)):
    emb = embeddings[idx, :]
    emb = emb.astype('float32')
    emb = Variable(torch.from_numpy(emb))
    emb = emb.cuda()
    rec = model.decodeSingle(emb)
    rec = rec.cpu()
    np_rec = rec.data.numpy()
    np_rec[np_rec > 0.0] = 255
    np_rec[np_rec < 0.0] = 0
    np_rec = np.reshape(np_rec, (1, 256 * 256))
    recreations[idx, :] = np_rec
    #np_rec = np.reshape(np_rec, (256, 256))
    #rfname = "sample_results/rec_"+str(idx+1)+".png"
    #scipy.misc.imsave(rfname, np_rec)

  return recreations

############################## LOADING STUFF ##############################

def loadNetwork(model):
  model = model.cuda()

  #checkpoint_to_load = 'model_checkpoints/archive/Cartesian/VAE_10k/checkpoint_test_2.pth.tar'
  checkpoint_to_load = 'model_checkpoints/archive/Cartesian/VAE_10k/checkpoint_train_83.pth.tar'
  #checkpoint_to_load = 'model_checkpoints/archive/Cartesian/VAE_repeat_init/checkpoint_train_262.pth.tar'

  if os.path.isfile(checkpoint_to_load):
    print("=> loading checkpoint '{}'".format(checkpoint_to_load))
    checkpoint = torch.load(checkpoint_to_load)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(checkpoint_to_load, checkpoint['epoch']))
  else:
    print("=> no checkpoint found at '{}'".format(checkpoint_to_load))

  n_parameters = sum([p.data.nelement() for p in model.parameters()])
  print('  + Number of params: {}'.format(n_parameters))

  return model

def loadScansFromBag(bag_file):
  bag = rosbag.Bag(bag_file)
  raw_scans = []
  for topic, msg, t in bag.read_messages(topics=['laser', '/laser', 'Cobot/Laser', '/Cobot/Laser']):
    raw_scans.append(np.asarray(msg.ranges))
  bag.close()
  np_raw_scans = np.empty([len(raw_scans), raw_scans[0].size])
  for i in range(len(raw_scans)):
    np_raw_scans[i, :] = raw_scans[i]
  return np_raw_scans

############################## MAIN ##############################

def main():

  bag_file = ""
  mode = -1
  if len(sys.argv) < 2:
    print("No mode selected. Usage: python testing_and_eval.py <bagfile> <mode>")
    return
  else:
    bag_file = sys.argv[1]
    mode = int(sys.argv[2])

  #TODO: pass in arg for which network to load
  model = VAE()
  #model = SimNet()
  model = loadNetwork(model)

  #TODO: pass in args for which dataset to create
  _, test_set = cptn.CurateTrainTest()
  if mode == 0: #For side by side playback of a deployment
    print("Running embedded view experiment")
    print("Writing downsampled inputs")
    #writeDownsampledInputs(test_set)
    print("Generating recreations")
    recreations = generateScanRecreationsFromData(test_set, model)
    print("Writing recreations to text")
    writeRecreationsToText(recreations)

  elif mode == 1: #For demonstrating continuity of embedding space
    print("Running embedding interpolation experiment")
    print("Generating embeddings")
    embeddings = generateInterpolatedEmbeddings(test_set, model)
    print("Generating recreations")
    recreations = generateScanRecreationsFromEmbeddings(embeddings, model)
    print("Writing recreations to text")
    writeRecreationsToText(recreations)

  elif mode == 2: #For checking how the dim-wise variance changes between raw, embedded, and recreated spaces
    print("Running dimension-wise variance experiment")
    print("Generating embeddings")
    embeddings = generateEmbeddings(test_set, model)
    print("Generating recreations")
    recreations = generateScanRecreationsFromData(test_set, model)
    print("Loading raw scans")
    raw_scans = loadScansFromBag(bag_file)

    print("Calulating variances")
    embedded_dimwise_vars = computeDimWiseVariance(embeddings)
    recreated_dimwise_vars = computeDimWiseVariance(recreations)
    raw_dimwise_vars = computeDimWiseVariance(raw_scans)

    plt.hist([embedded_dimwise_vars], bins=[0.0, 0.0000025, 0.000005,  0.00001, 0.00002, 0.00004, 0.00008, 0.00016, 0.00032, 0.00064])
    plt.show()

    plt.hist([recreated_dimwise_vars], bins=[0.0, 0.0000025, 0.000005,  0.00001, 0.00002, 0.00004, 0.00008, 0.00016, 0.00032, 0.00064])
    plt.show()

    plt.hist([raw_dimwise_vars], bins=[0.0, 0.0000025, 0.000005,  0.00001, 0.00002, 0.00004, 0.00008, 0.00016, 0.00032, 0.00064])
    plt.show()

  elif mode == 3: #For qualitative analysis of embedded / recreated / raw trajectories
    print("Running trajectory analysis experiment")
    print("Generating embeddings")
    embeddings = generateEmbeddings(test_set, model)
    #print("Generating recreations")
    #recreations = generateScanRecreationsFromData(test_set, model)
    print("Loading raw scans")
    raw_scans = loadScansFromBag(bag_file)

    print("Computing data trajectories")
    embedding_trajectories = generateDataTrajectories(embeddings)
    #recreation_trajectories = generateDataTrajectories(recreations)
    raw_trajectories = generateDataTrajectories(raw_scans)

    vizDataTrajectories(embedding_trajectories)
    #vizDataTrajectories(recreation_trajectories)
    vizDataTrajectories(raw_trajectories)

  elif mode == 4:
    #TODO: anythin else I want to be able to do / save old methods
    print("undefined")

if __name__ == '__main__':
  main()
  print ("donzo")
