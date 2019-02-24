#!/usr/bin/env python2

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:35:43 2018

@author: samer

"""

from __future__ import print_function

import rospy
#from std_msgs.msg import string
#from std_msgs.msg import String
from gui_msgs.msg import LaserQueryImage


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
from network_definitions import VAE, SimNet
from scipy import spatial
from visdom import Visdom
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.decomposition import PCA, KernelPCA

############################## QUERY EXPERIMENTS ##############################

# WHOLE SCANS: noise-wise ordering is correct, noise/non-noise order is more or less preserved
# 1. VAE + SIMNET
# 2. FLIRT + SIMNET / RANSAC (whichever will work)
# 3. FALKO + SIMNET / RANSAC (whichever will work)
# 3b. FLIRT + FALKO + SIMNET
# 4. NAIVENET
# 5. RAW INPUT

# SUBSET SCANS: Get recall, precision for synthetically added subsets
# 1. VAE + SIMNET
# 2. VAE + SET REMAINDER TO 0
# 3. VAE + MONTE CARLO SUBSPACE
# 4. NAIVENET
# 5. RAW INPUT
# 6. FLIRT / FALKO (if they'll work)



############################## LOADING STUFF ##############################

def loadNetwork(model, checkpoint_to_load):
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

def loadEmbeddingDatabase():
  database_set = cptn.CurateQuerySet()
  global embeddings_database
  embeddings_database = generateEmbeddings(database_set)

def generateEmbeddings(database_set):
  vae.eval()
  database_len = len(database_set)
  input_scan, _ = test_set[0]
  input_scan = input_scan.unsqueeze(0)
  with torch.no_grad():
    input_scan = Variable(input_scan)
  input_scan = input_scan.cuda()
  _, emb, _ = vae(input_scan)
  emb = emb.cpu()
  np_emb = emb.data.numpy()
  embeddings = np.empty([database_len, np_emb.size])
  for idx in range(database_len):
    if idx % 100 == 0:
      print(idx+" / "+database_len)

    input_scan, _ = test_set[idx]
    input_scan = input_scan.unsqueeze(0)
    with torch.no_grad():
      input_scan = Variable(input_scan)
    input_scan = input_scan.cuda()
    _, emb, _ = vae(input_scan)

    emb = emb.cpu()
    np_emb = emb.data.numpy()
    embeddings[idx, :] = np_emb

  return embeddings

def generateNetQueryEmbedding(msg):
  vae.eval()
  #TODO: get input_scan from msg
  width = msg.width
  height = msg.height
  #TODO: test
  input_scan = np.asarray(msg.row_major_bitmap)

  rfname = "sample_results/rec_"+str(idx+1)+".png"
  scipy.misc.imsave(rfname, np_rec)

  input_scan = transformForTorch(input_scan)
  with torch.no_grad():
    input_scan = Variable(input_scan)
  input_scan = input_scan.cuda()
  _, emb, _ = vae(input_scan)
  emb = emb.cpu()
  np_emb = emb.data.numpy()
  return np_emb

def transformForTorch(scan):
  to_tensor = transforms.ToTensor()
  normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  transform = transforms.Compose([to_tensor, normalize])
  scan = transform(scan)
  scan = scan[0, :, :]
  scan = scan.unsqueeze(0)
  return scan

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

############################## DISTANCE ##############################

def computeSubspaceDistance(a, b, basis_vectors, weights):
  diff = a - b
  subspace_distance = 0
  for i in range(len(weights)):
    new_dist = abs(np.dot(diff, basis_vectors[i, :]) / np.linalg.norm(basis_vectors[i, :]))
    subspace_distance = subspace_distance + weights[i] * new_dist
  return subspace_distance

############################## TOP K ##############################

def KNNOnEmbeddings(database, queries):
    K = 300

    query_results = []
    for idx in range(len(queries)):
      print(idx)
      x = queries[idx]
      query_result = []
      for idt in range(len(database)):
        if idt % 1000 == 0:
          print(str(idt)+"/"+str(len(database)))
        y = database[idt]
        y = np.asarray(y)
        dissimilarity = spatial.distance.cosine(y, x)
        #dissimilarity = np.linalg.norm(y - x)
        single_scan = []
        if len(query_result) < K:
          single_scan.append(idt)
          single_scan.append(dissimilarity)
          query_result.append(single_scan)
        else:
          if dissimilarity < query_result[0][1]:
            single_scan.append(idt)
            single_scan.append(dissimilarity)
            query_result[0] = single_scan
            # reverse sort based on similarity
            query_result.sort(key = lambda k: (k[1]), reverse=True)

      query_results.append(query_result)

    return query_results

def SubspaceKNNOnEmbeddings(database, queries, basis_vectors, weights):
    K = 10

    query_results = []
    for idx in range(len(queries)):
      print(idx)
      x = queries[idx]
      query_result = []
      for idt in range(len(database)):
        if idt % 1000 == 0:
          print(idt)
        y = database[idt]
        y = np.asarray(y)
        dissimilarity = computeSubspaceDistance(x, y, basis_vectors, weights)
        single_scan = []
        if len(query_result) < K:
          single_scan.append(idt)
          single_scan.append(dissimilarity)
          query_result.append(single_scan)
        else:
          if dissimilarity < query_result[0][1]:
            single_scan.append(idt)
            single_scan.append(dissimilarity)
            query_result[0] = single_scan
            # reverse sort based on similarity
            query_result.sort(key = lambda k: (k[1]), reverse=True)

      query_results.append(query_result)

    return query_results

def SimNetOnEmbeddings():
  print("soon")

############################## QUERY TYPES ##############################

def simpleQuery(query_embeddings):

  npqueryembed = np.zeros((len(query_embeddings), len(query_embeddings[0])))

  final_query_embeddings = []
  embedding_sum = np.zeros(len(query_embeddings[0]))
  print("number of embedding examples: "+str(len(query_embeddings)))
  for i in range(len(query_embeddings)):
    embedding_sum = embedding_sum + np.asarray(query_embeddings[i])
    npqueryembed[i, :] = np.asarray(query_embeddings[i])
  embedding_mean = np.true_divide(embedding_sum, len(query_embeddings))
  final_query_embeddings.append(embedding_mean)

  #print(embeddings_database[0])
  #print(final_query_embeddings[0])

  query_results = KNNOnEmbeddings(embeddings_database, final_query_embeddings)

  for i in range(len(query_results)):
    print("QUERY: "+str(i))
    for j in range(len(query_results[i])):
      print(query_results[i][j])


def advancedQuery():


  print("finding query mean")
  print(len(query_embeddings))
  print(len(query_embeddings[0]))

  npqueryembed = np.zeros((len(query_embeddings), len(query_embeddings[0])))

  final_query_embeddings = []
  embedding_sum = np.zeros(len(query_embeddings[0]))
  print("number of embedding examples: "+str(len(query_embeddings)))
  for i in range(len(query_embeddings)):
    embedding_sum = embedding_sum + np.asarray(query_embeddings[i])
    npqueryembed[i, :] = np.asarray(query_embeddings[i])
  embedding_mean = np.true_divide(embedding_sum, len(query_embeddings))
  final_query_embeddings.append(embedding_mean)


  kappa = 5
  #basis_vectors = np.zeros((len(final_query_embeddings[0]), 5))
  basis_vectors = np.zeros((5, len(final_query_embeddings[0])))
  weights = np.ones(kappa)
  #TODO: use xplained variance to weight each dimension in subspace distance calculation?
  #weights = np.zeros(kappa)
  if len(query_embeddings) > 1:
    print("PCA")
    pca = PCA()
    pca.fit(npqueryembed)
    components = pca.components_
    print(components)
    print(components.shape)
    for i in range(kappa):
      basis_vectors[i, :] = components[len(final_query_embeddings[0]) - 1 - i, :]

  print("KNN")

  #query_results = KNNOnEmbeddings(database_embeddings, final_query_embeddings)
  query_results = SubspaceKNNOnEmbeddings(database_embeddings, final_query_embeddings, basis_vectors, weights)

  for i in range(len(query_results)):
    print("QUERY: "+str(i))
    for j in range(len(query_results[i])):
      print(query_results[i][j])

def NetQuery(q):
  K = 300

  query_results = []
  for idx in range(embeddings_database.size(0)):
    if idx % 1000 == 0:
      print(str(idx)+" / "+str(embeddings_database.size()))
    db_item = database[idx, :]
    similarity = simnet(db_item, q)
    single_scan = []
    if len(query_results) < K:
      single_scan.append(idx)
      single_scan.append(similarity)
      query_result.append(single_scan)
    else:
      if similarity > query_results[0][1]:
        single_scan.append(idx)
        single_scan.append(similarity)
        query_result[0] = single_scan
        # reverse sort based on similarity
        query_results.sort(key = lambda k: (k[1]), reverse=True)

  return query_results

def queryCallback(msg):
  #TODO: enumerate query processing types
  if True:
    q = generateNetQueryEmbedding(msg)
    query_results = NetQuery(q)
    publishQueryResults(query_results)
  else:
    print("not yet")

def publishQueryResults():
  

############################## MAIN ##############################

def main():
  #TODO: take network versions, query types, and datasets as command line args
  vae = VAE()
  checkpoint_to_load = 'model_checkpoints/archive/Cartesian/VAE_10k/checkpoint_test_83.pth.tar'
  vae = loadNetwork(vae, checkpoint_to_load)
  loadEmbeddingDatabase(vae)
  print("database size:")
  print(len(embeddings_database))

  global simnet = SimNet()
  checkpoint_to_load = 'model_checkpoints/archive/Cartesian/SimNet_10k/checkpoint_test_5.pth.tar'
  global simnet = loadNetwork(simnet, checkpoint_to_load)

  rospy.init_node('QueryManager', anonymous=True)
  rospy.Subscriber('LaserQueryImage', QueryImage, queryCallback)
  #global query_results_publisher = rospy.Publisher('', messagetype, queue_size=1)

  print("Ready to answer queries")

  rospy.spin()

if __name__ == '__main__':
  main()
  print ("donzo")
