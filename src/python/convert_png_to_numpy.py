#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:55:36 2018

@author: samer

"""

import cv2
import torch
import random
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import utils


class LaserDataset(Dataset):
  """Laser dataset."""

  def __init__(self, root_dir, bag_dates_scan_lengths, transform=None):
      """
      Args:
          bag_dates... ((N, 2) array): Pairs of dates and scan lengths.
          root_dir (string): Directory with all the images.
          transform (callable, optional): Optional transform to be applied
              on a sample.
      """
      self.root_dir = root_dir
      self.transform = transform
      self.record = []
      for i in range(len(bag_dates_scan_lengths)): # pairs ((N, 2) array) of dates and scans
        bag_date = bag_dates_scan_lengths[i][0]
        for j in range(1, bag_dates_scan_lengths[i][1] + 1):
          self.record.append(root_dir+bag_date+"/"+bag_date+"_"+str(j)+".png")

  def PNGtoNPA(self, file_name):
    im = cv2.imread(file_name)
    return im
  
  def __len__(self):
    return len(self.record)

  def __getitem__(self, idx):
    #TODO: make better later - idx means nothing atm
    a_idx = random.randint(0, len(self.record)-1)
    aprime_idx = a_idx - 1
    if a_idx == 0:
      aprime_idx = 1
    
    b_idx = random.randint(0, len(self.record)-1)
    
    sample_a = self.PNGtoNPA(self.record[a_idx])
    sample_aprime = self.PNGtoNPA(self.record[aprime_idx])
    sample_b = self.PNGtoNPA(self.record[b_idx])
    if self.transform:
      sample_a = self.transform(sample_a)
      sample_aprime = self.transform(sample_aprime)
      sample_b = self.transform(sample_b)

    sample_a = sample_a[0, :, :]
    sample_a = sample_a.unsqueeze(0)
    sample_aprime = sample_aprime[0, :, :]
    sample_aprime = sample_aprime.unsqueeze(0)
    sample_b = sample_b[0, :, :]
    sample_b = sample_b.unsqueeze(0)
 
    return sample_a, sample_aprime, sample_b
  
def CurateTrainTest():
  to_tensor = transforms.ToTensor()
  normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  transform = transforms.Compose([to_tensor, normalize])

# NOTE: MASTER LIST. DO NOT ALTER. ONLY COPY.
#  bdsl_train = [["2016-08-17-12-48-06", 403], 
#                ["2016-08-17-12-56-32", 544], 
#                ["2016-08-17-12-57-16", 835],
#                ["2016-08-17-12-57-54", 672],
#                ["2016-08-17-12-58-32", 1191],
#                ["2016-04-17-20-32-59", 2723],
#                ["2016-04-17-20-34-16", 2332],
#                ["2016-08-17-13-08-32", 2649],
#                ["2016-08-17-13-16-56", 2832],
#                ["2016-08-17-13-18-08", 3283],
#                ["2016-02-16-15-46-43", 8971],
#                ["2016-02-16-16-01-46", 9799],
#                ["2016-02-16-16-17-06", 7251],
#                ["2016-02-16-21-18-05", 6498],
#                ["2016-02-16-21-20-53", 9099]]

  bdsl_train = [["2016-08-17-12-48-06", 403], 
                ["2016-08-17-12-56-32", 544], 
                ["2016-08-17-12-57-16", 835],
                ["2016-08-17-12-57-54", 672],
                ["2016-08-17-12-58-32", 1191],
                ["2016-04-17-20-32-59", 2723],
                ["2016-04-17-20-34-16", 2332],
                ["2016-08-17-13-08-32", 2649],
                ["2016-08-17-13-16-56", 2832],
                ["2016-02-16-15-46-43", 8971],
                ["2016-02-16-16-01-46", 9799],
                ["2016-02-16-16-17-06", 7251],
                ["2016-02-16-21-18-05", 6498],
                ["2016-02-16-21-20-53", 9099]]

#  bdsl_train = [["2016-08-17-12-48-06", 160]] 


  laser_dataset_train = LaserDataset('../../laser_images/', bdsl_train, transform=transform)

  #TODO: better way to differentiate training and testing sets
  bdsl_test = [["2016-08-17-13-18-08", 3283]]
  laser_dataset_test = LaserDataset('../../laser_images/', bdsl_test, transform=transform)

  print("Training dataset size: ")
  print(len(laser_dataset_train))

  train_loader = torch.utils.data.DataLoader(laser_dataset_train, batch_size=16, shuffle=True)
  test_loader = torch.utils.data.DataLoader(laser_dataset_test, batch_size=4, shuffle=False)

  return train_loader, test_loader

def SpecialTestSet():
  to_tensor = transforms.ToTensor()
  normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  transform = transforms.Compose([to_tensor, normalize])

  #TODO: new way to choose subset of data for testing
  bdsl_test = [["hand_picked", TBD]]
  laser_dataset_test = LaserDataset('../../laser_images/', bdsl_test, transform=transform)

  test_loader = torch.utils.data.DataLoader(laser_dataset_test, batch_size=4, shuffle=False)

  return test_loader
  
  
  
  
