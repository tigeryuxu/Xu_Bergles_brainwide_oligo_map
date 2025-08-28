# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:10:03 2020

@author: tiger
"""


import torchvision.transforms.functional as TF
import random
from torch.utils import data
import torch
import time
import numpy as np

import scipy
import math


""" Calculate Jaccard on the GPU """
def jacc_eval_GPU_torch(output, truth, ax_labels=-1, argmax_truth=1):
      output = torch.argmax(output,axis=1)
      intersection = torch.sum(torch.sum(output * truth, axis=ax_labels),axis=ax_labels)
      union = torch.sum(torch.sum(torch.add(output, truth)>= 1, axis=ax_labels),axis=ax_labels) + 0.0000001
      jaccard = torch.mean(intersection / union)  # find mean of jaccard over all slices        
      return jaccard

""" Define transforms"""
# import torchio as tio
# from torchio.transforms import (
#     RescaleIntensity,
#     RandomFlip,
#     RandomAffine,
#     RandomElasticDeformation,
#     RandomMotion,
#     RandomBiasField,
#     RandomBlur,
#     RandomNoise,
#     Interpolation,
#     Compose
# )
#from torchio import Image, Subject, ImagesDataset

# def initialize_transforms(p=0.5):
#      transforms = [
#            tio.RandomFlip(axes = 0, flip_probability = 0.5, p = p, seed = None),
           
#            tio.RandomAffine(scales=(0.9, 1.1), degrees=(10), isotropic=False,
#                         default_pad_value='otsu', image_interpolation='linear',
#                         p = p, seed=None),
           
#            # *** SLOWS DOWN DATALOADER ***
#            #RandomElasticDeformation(num_control_points = 7, max_displacement = 7.5,
#            #                         locked_borders = 2, image_interpolation = Interpolation.LINEAR,
#            #                         p = 0.5, seed = None),
#            tio.RandomMotion(degrees = 10, translation = 10, num_transforms = 2, image_interpolation = 'linear',
#                         p = p, seed = None),
           
#            #tio.RandomBiasField(coefficients=0.5, order = 3, p = p, seed = None),
           
#            tio.RandomBlur(std = (0, 4), p = p, seed=None),
           
#            tio.RandomNoise(mean = 0, std = (0, 0.25), p = p, seed = None),
#            #RescaleIntensity((0, 255))
           
#      ]
#      transform = tio.Compose(transforms)
#      return transform


def initialize_transforms_simple(p=0.5):
     transforms = [
           tio.RandomFlip(axes = (0, 1, 2), flip_probability = 0.5),
           
           tio.RandomAffine(scales=(0.9, 1.1), degrees=(10), image_interpolation='linear'),
           
           
           #tio.RandomAnisotropy(axes=0, downsampling=(1, 2)),
           

           #tio.RandomMotion(degrees = 2, translation = 2, num_transforms = 3, image_interpolation = 'linear'),
           
           #tio.RandomBiasField(coefficients=0.5, order = 3),
           
           tio.RandomBlur(std = (0, 2)),
           
           tio.RandomNoise(mean = 0, std = (0, 0.25)),
           #tio.RescaleIntensity((0, 255))
           
     ]
     transform = tio.Compose(transforms)
     return transform



""" Do pre-processing on GPU
          ***can't do augmentation/transforms here because of CPU requirement for torchio

"""
def transfer_to_GPU(X, Y, device, mean, std, transforms = 0):
     """ Put these at beginning later """
     mean = torch.tensor(mean, dtype = torch.float, device=device, requires_grad=False)
     std = torch.tensor(std, dtype = torch.float, device=device, requires_grad=False)
     
     """ Convert to Tensor """
     inputs = torch.tensor(X, dtype = torch.float, device=device, requires_grad=False)
     labels = torch.tensor(Y, dtype = torch.long, device=device, requires_grad=False)           

     """ Normalization """
     inputs = (inputs - mean)/std
                
     """ Expand dims """
     if len(inputs.shape) < 5:   ### only if necessary
         inputs = inputs.unsqueeze(1)   

     return inputs, labels






""" Load data directly from tiffs """
import tifffile as tifffile
class Dataset_tiffs(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, examples, mean, std, sp_weight_bool=0, transforms=0):
        'Initialization'
        #self.labels = labels
        self.list_IDs = list_IDs
        self.examples = examples
        self.transforms = transforms
        self.mean = mean
        self.std = std
        self.sp_weight_bool = sp_weight_bool

  def apply_transforms(self, image, labels):
        #inputs = np.asarray(image, dtype=np.float32)
        # inputs = np.expand_dims(image, axis=0)
        # labels = np.expand_dims(labels, axis=0)

 
        # inputs = torch.tensor(inputs, dtype = torch.float,requires_grad=False)
        # labels = torch.tensor(labels, dtype = torch.long, requires_grad=False)         
 
        # subject_a = tio.Subject(
        #         one_image=tio.ScalarImage(tensor=inputs),   # *** must be tensors!!!
        #         a_segmentation=tio.LabelMap(tensor=labels))
          
        # subjects_list = [subject_a]

        # subjects_dataset = tio.SubjectsDataset(subjects_list, transform=self.transforms)
        # subject_sample = subjects_dataset[0]
          
          
        # X = subject_sample['one_image']['data'].numpy()
        # Y = subject_sample['a_segmentation']['data'].numpy()
        
        
        
        
        """ As pure numpy"""
        inputs = np.expand_dims(image, axis=0)
        labels = np.expand_dims(labels, axis=0)

 
        #inputs = torch.tensor(inputs, dtype = torch.float,requires_grad=False)
        #labels = torch.tensor(labels, dtype = torch.long, requires_grad=False)         
 
        subject_a = tio.Subject(
                one_image=tio.ScalarImage(tensor=inputs),   # *** must be tensors!!!
                a_segmentation=tio.LabelMap(tensor=labels))
          
        subjects_list = [subject_a]

        subjects_dataset = tio.SubjectsDataset(subjects_list, transform=self.transforms)
        subject_sample = subjects_dataset[0]
          
          
        X = subject_sample['one_image']['data']
        Y = subject_sample['a_segmentation']['data']        
        
        
        
        
        
        
        
        
        return X[0], Y[0]

  def create_spatial_weight_mat(self, labels, edgeFalloff=10,background=0.01,approximate=True):
       
         if approximate:   # does chebyshev
             dist1 = scipy.ndimage.distance_transform_cdt(labels)
             dist2 = scipy.ndimage.distance_transform_cdt(np.where(labels>0,0,1))    # sets everything in the middle of the OBJECT to be 0
                     
         else:   # does euclidean
             dist1 = scipy.ndimage.distance_transform_edt(labels, sampling=[1,1,1])
             dist2 = scipy.ndimage.distance_transform_edt(np.where(labels>0,0,1), sampling=[1,1,1])
             
         """ DO CLASS WEIGHTING instead of spatial weighting WITHIN the object """
         dist1[dist1 > 0] = 0.5
     
         dist = dist1+dist2
         attention = math.e**(1-dist/edgeFalloff) + background   # adds background so no loses go to zero
         attention /= np.average(attention)
         return np.reshape(attention,labels.shape)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        #y = self.labels[ID]

 
        input_name = self.examples[ID]['input']
        truth_name = self.examples[ID]['truth']

        X = tifffile.imread(input_name)
        
        # print(X.shape)
        import os
        red_name = self.examples[ID]['autofluor']
        # myelin_name = self.examples[ID]['myelin']
        if os.path.exists(red_name):
            # X2 = tifffile.imread(red_name)
        
            # X = np.expand_dims(X, axis=0)
            # X2 = np.expand_dims(X2, axis=0)
            
            # X = np.concatenate((X, X2))
            
            
            
            ### If want to only do autofluor instead
            X = tifffile.imread(red_name)
            

        
        # zzz
        #X = np.expand_dims(X, axis=0)
        Y = tifffile.imread(truth_name)
        Y[Y > 0] = 1
        #Y = np.expand_dims(Y, axis=0)
        
        
        """ Pytorch does not support uint16, so have to cast to int16 """
        
        if X.dtype == np.uint16:
            X = np.asarray(X, dtype=np.int16)
        if Y.dtype == np.uint16:
            Y = np.asarray(Y, dtype=np.int16)


        
        
        """ Get spatial weight matrix """
        if self.sp_weight_bool:
             spatial_weight = self.create_spatial_weight_mat(Y)
             
        else:
             spatial_weight = []
             
             
        """ Do normalization here??? """
        #X  = (X  - self.mean)/self.std


        """ Transforms """
        if self.transforms:
              X, Y = self.apply_transforms(X, Y)  
        
        
        
        
        
        # """ If want to do lr_finder """
        # X = np.asarray(X, dtype=np.float32)
        # X = (X - self.mean)/self.std
                    
        # """ Expand dims """
        # #X = inputs.unsqueeze(0)  
        # X = np.expand_dims(X, axis=0)
        # #Y = labels
        # X = torch.tensor(X, dtype = torch.float, requires_grad=False)
        # Y = torch.tensor(Y, dtype = torch.long, requires_grad=False)            
        
        return X, Y, spatial_weight


