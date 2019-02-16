#!/usr/bin/env python2

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:35:43 2018

@author: samer

"""

#TODO: refactor into separate files for each network, and
#      separate files for nominal training versus special cases / debugging

from __future__ import print_function

import rospy
#from std_msgs.msg import string
from std_msgs.msg import String


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
from network_definitions import VAE, TripletNet
from scipy import spatial
from visdom import Visdom
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.decomposition import PCA, KernelPCA

############################## LOADING STUFF ##############################

#TODO: might not need
def loadNetwork():
  model = net3.Net3()
  model = model.cuda()
  tnet = net3.TripletNet(model)
  tnet = tnet.cuda()

  checkpoint_to_load = 'model_checkpoints/current/most_recent.pth.tar'

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

  return tnet

def loadEmbeddingDatabase():
  database_set = cptn.SpecialQuerySet()
  global dn, dr
  dn, dr = database_set.getSpecificItem(0)
  global embeddings_database
  #embeddings_database = generateEmbeddings(database_set)
  embeddings_database = []

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

def NetQuery():
  print("not defined yet")

def queryCallback(msg):
  rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.data)
  if msg.data == "FO":
    query_set = cptn.FeatureOnlyTestSet()
    #qn, qr = query_set.getSpecificItem(0)
    #d_scan = dn[0, 0, :]
    #q_scan = qn[0, 0, :]
    #print("difference")
    #print(d_scan - q_scan)
    query_embeddings = generateEmbeddings(query_set)
    global embeddings_database
    embeddings_database = query_embeddings
    visEmbeddingDecoder()

    diff = []
    for j in range(len(query_embeddings[0])):
      diff.append(query_embeddings[1][j] - query_embeddings[0][j])
    query_embeddings.append(diff)

    #for i in range(len(query_embeddings)):
    print(query_embeddings[0])
#    for i in range(1):
#      for j in range(len(query_embeddings[i])):
#        print(str(query_embeddings[i][j])+", ")
#      print("\n\n")


    #############simpleQuery(query_embeddings)
  elif msg.data == "MC":
    query_set = cptn.MonteCarloTestSet()
    query_embeddings = generateEmbeddings(query_set)
    advancedQuery()

############################## MAIN ##############################

def main():
  global tnet 
  tnet = loadNetwork()
  loadEmbeddingDatabase()

  #NOTE: uncomment to turn on embedding interpolation - saved in recreations.txt
  #visEmbeddingDecoder()

  print("database size:")
  print(len(embeddings_database))

  rospy.init_node('QueryManager', anonymous=True)
  rospy.Subscriber("DataDirectory", String, queryCallback)

  print("Ready to answer queries")

  rospy.spin()

if __name__ == '__main__':
  main()
  print ("donzo")
