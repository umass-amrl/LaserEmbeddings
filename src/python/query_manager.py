#!/usr/bin/env python2

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:35:43 2018

@author: samer

"""

from __future__ import print_function

import rospy
import std_msgs.msg
#TODO: why doesn't this work???
#from gui_msgs.msg import QueryImageMsg
#from gui_msgs.msg import QueryResultsMsg
from _QueryImageMsg import QueryImageMsg
from _QueryResultsMsg import QueryResultsMsg


import os
#import copy
import torch
import shutil
import argparse
import scipy.misc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import convert_png_to_numpy as cptn
import torch.backends.cudnn as cudnn
from network_definitions import VAE, SimNet
from network_definitions import SimNetTrainer
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

# BONUS: embeddings work for other perception-related tasks

############################## GLOBALS ##############################

global vae
global simnet
global embeddings_database
global query_results_publisher

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

def loadEmbeddingDatabase(model):
  database_set = cptn.CurateQuerySet()
  global embeddings_database
  embeddings_database = generateEmbeddings(model, database_set)

def generateEmbeddings(model, database_set):
  model.eval()
  model = model.cuda()
  database_len = len(database_set)
  input_scan, _ = database_set[0]
  input_scan = input_scan.unsqueeze(0)
  with torch.no_grad():
    input_scan = Variable(input_scan)
  input_scan = input_scan.cuda()
  _, emb, _ = model(input_scan)
  emb = emb.cpu()
  np_emb = emb.data.numpy()
  embeddings = np.empty([database_len, np_emb.size])
  #for idx in range(database_len):
  for idx in range(1001):
    if idx % 100 == 0:
      print(str(idx)+" / "+str(database_len))

    input_scan, _ = database_set[idx]
    #if idx == 0:
    #  print(input_scan.shape)
    #  rfname = "sample_results/first_emb_input.png"
    #  scipy.misc.imsave(rfname, input_scan.squeeze(0))
    input_scan = input_scan.unsqueeze(0)
    with torch.no_grad():
      input_scan = Variable(input_scan)
    input_scan = input_scan.cuda()
    #print(input_scan.shape)
    _, emb, _ = model(input_scan)

    emb = emb.cpu()
    np_emb = emb.data.numpy()
    embeddings[idx, :] = np_emb

  return embeddings

def generateNetQueryEmbedding(msg):
  vae.eval()
  width = msg.width
  height = msg.height
  input_scan = np.asarray(msg.row_major_bitmap, dtype='uint8')
  input_scan = np.reshape(input_scan, (256, 256, 1))
  #rfname = "sample_results/query_input.png"
  #scipy.misc.imsave(rfname, input_scan.squeeze(2))
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
  scan = scan.unsqueeze(0)
  return scan

#def induceTranspose(input_scan_img):
#  input_scan_img[0, :, :] = np.flip(input_scan_img[0, :, :], 0)
#  input_scan_img[0, :, :] = np.flip(input_scan_img[0, :, :], 1)
#  return input_scan_img


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
  global simnet
  simnet.eval()
  simnet = simnet.cuda()
  with torch.no_grad():
    q = Variable(torch.from_numpy(q))
  q = q.cuda()
  query_results = []
  K = 300
  #for idx in range(len(embeddings_database)):
  for idx in range(1001):
    if idx % 1000 == 0:
      print(str(idx)+" / "+str(len(embeddings_database)))
    db_item = embeddings_database[idx, :]
    db_item = np.expand_dims(db_item, axis=0)
    db_item = db_item.astype('float32')
    with torch.no_grad():
      db_item = Variable(torch.from_numpy(db_item))
    db_item = db_item.cuda()
    similarity, sim_test = simnet(db_item, db_item, q)
    similarity = similarity.cpu().data.numpy()
    sim_test = sim_test.cpu().data.numpy()
    single_scan = []
    if len(query_results) < K:
      single_scan.append(idx)
      single_scan.append(similarity)
      query_results.append(single_scan)
    else:
      if similarity > query_results[0][1]:
        single_scan.append(idx)
        single_scan.append(similarity)
        query_results[0] = single_scan
        # reverse sort based on similarity
        query_results.sort(key = lambda k: (k[1]), reverse=True)

  return query_results

def queryCallback(msg):
  #TODO: enumerate query processing types
  if True:
    q = generateNetQueryEmbedding(msg)
    print("do the query")
    query_results = NetQuery(q)
    publishQueryResults(query_results)
  else:
    print("not yet")

def publishQueryResults(query_results):
  #header = std_msgs.msg.Header()
  #header.stamp = rospy.Time.now()
  query_results_msg = QueryResultsMsg()
  #query_results_msg.header = header
  query_results_msg.timestamp = rospy.Time.now()
  query_results_msg.k = 300
  scan_ids = []
  for i in range(len(query_results)):
    scan_ids.append(query_results[i][0])
  query_results_msg.top_k_scan_ids = scan_ids
  #global query_results_publisher.publish(query_results_msg)
  query_results_publisher.publish(query_results_msg)

############################## MAIN ##############################

def main():
  #TODO: take network versions, query types, and datasets as command line args
  global vae
  vae = VAE()
  checkpoint_to_load = 'model_checkpoints/archive/Cartesian/VAE_10k/checkpoint_test_83.pth.tar'
  vae = loadNetwork(vae, checkpoint_to_load)
  loadEmbeddingDatabase(vae)
  print("database size:")
  print(len(embeddings_database))

  global simnet
  #TODO: verify underlying simnets are identical
  #simnet = SimNet()
  subnet = SimNet()
  simnet = SimNetTrainer(subnet)
  checkpoint_to_load = 'model_checkpoints/archive/Cartesian/SimNet_10k/checkpoint_test_5.pth.tar'
  simnet = loadNetwork(simnet, checkpoint_to_load)

  rospy.init_node('QueryManager', anonymous=True)
  rospy.Subscriber('LaserQueryImage', QueryImageMsg, queryCallback)
  global query_results_publisher
  query_results_publisher = rospy.Publisher('QueryResults', QueryResultsMsg, queue_size=1)

  print("Ready to answer queries")

  rospy.spin()

if __name__ == '__main__':
  main()
  print ("donzo")
