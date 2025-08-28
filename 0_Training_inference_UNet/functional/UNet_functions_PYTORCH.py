#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:10:34 2020

@author: user
"""


import torch
import numpy as np
from functional.data_functions_CLEANED import *

import matplotlib.pyplot as plt

#from PYTORCH_dataloader import *

def plot_max(im, ax=0, plot=1):
     max_im = np.amax(im, axis=ax)
     if plot:
         plt.figure(); plt.imshow(max_im)
     
     return max_im

def normalize(im, mean, std):
    return (im - mean)/std
    


""" Do pre-processing on GPU
          ***this is from the PYTORCH_dataloader.py file
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
     inputs = inputs.unsqueeze(1)   

     return inputs, labels



    
""" Perform inference by splitting input volume into subparts """
def UNet_inference_by_subparts_PYTORCH(unet, device, input_im, overlap_percent, quad_size, quad_depth, mean_arr, std_arr, skip_top=0, num_truth_class=1, batch_size=8):
     im_size = np.shape(input_im);
     width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
        
     segmentation = np.zeros([depth_im, width, height])
     total_blocks = 0;
     all_xyz = []                                               
     
     
     batch_x = []
     batch_coords = []
    
        
     for x in range(0, width + quad_size, round(quad_size - quad_size * overlap_percent)):
          if x + quad_size > width:
               difference = (x + quad_size) - width
               x = x - difference
                    
          for y in range(0, height + quad_size, round(quad_size - quad_size * overlap_percent)):
               
               if y + quad_size > height:
                    difference = (y + quad_size) - height
                    y = y - difference
               
               for z in range(0, depth_im + quad_depth, round(quad_depth - quad_depth * overlap_percent)):
                   #batch_x = []; batch_y = [];
         
                   if z + quad_depth > depth_im:
                        difference = (z + quad_depth) - depth_im
                        z = z - difference
                   
                       
                   """ Check if repeated """
                   skip = 0
                   for coord in all_xyz:
                        if coord == [x,y,z]:
                             skip = 1
                             break                      
                   if skip:  continue
                        
                   all_xyz.append([x, y, z])
                   
                   quad_intensity = input_im[z:z + quad_depth, x:x + quad_size, y:y + quad_size];  
                                     

                   """ Normalization """
                   ### maybe needs to be int16???
                   quad_intensity = np.asarray(quad_intensity, dtype=np.int16)
                   #quad_intensity = normalize(quad_intensity, mean_arr, std_arr)

                   total_blocks += 1
                   """ Analyze """
                   """ set inputs and truth """
                   if len(batch_x) < batch_size:
                       quad_intensity = np.expand_dims(quad_intensity, axis=0)
                       batch_x.append(quad_intensity)
                       batch_coords.append([z, x, y])
                       
                       continue
                   elif batch_size == 1 or len(batch_x) >= batch_size:
                       quad_intensity = np.expand_dims(quad_intensity, axis=0)
                       batch_x.append(quad_intensity)
                       batch_coords.append([z, x, y])                       
                       
                       batch_x = np.vstack(batch_x)
                       batch_coords = np.vstack(batch_coords)
                                              
                       
                   #quad_intensity = np.expand_dims(quad_intensity, axis=-1)
        
                   
                   #batch_x = quad_intensity
                   #batch_x = np.moveaxis(batch_x, -1, 0)
                   #batch_x = np.expand_dims(batch_x, axis=0)
                   
                   
                   batch_y = np.zeros([batch_size, num_truth_class, quad_depth, quad_size, quad_size])
                   
                   """ Call transfer to GPU """
                   inputs, labels = transfer_to_GPU(batch_x, batch_y, device, mean_arr, std_arr)
                   
                   

               
                   """ Convert to Tensor """
                   #inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)
                   #labels_val = torch.tensor(batch_y, dtype = torch.long, device=device, requires_grad=False)
         
                   # forward pass to check validation
                   output_val = unet(inputs)

                   """ Convert back to cpu """                                      
                   output_tile = output_val.cpu().data.numpy()            
                   output_tile = np.moveaxis(output_tile, 1, -1)
                   #seg_train = np.argmax(output_tile[0], axis=-1)  
                   
                   seg_train = np.argmax(output_tile, axis=-1)
                    

                   ### add in by batch
                   for id_b, cleaned_seg in enumerate(seg_train):
                       coords = batch_coords[id_b]
                       z = coords[0]; x = coords[1]; y = coords[2]
                       segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = cleaned_seg + segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
         

                   ### reset batch
                   batch_x = []
                   batch_coords = []                   
                   
                   """ Clean segmentation by removing objects on the edge """
                   #cleaned_seg = seg_train
                   # if skip_top and z == 0:
                   #      #print('skip top')
                   #      cleaned_seg = clean_edges(seg_train, extra_z=1, extra_xy=3, skip_top=skip_top)                                             
                   # else:
                   #      cleaned_seg = clean_edges(seg_train, extra_z=1, extra_xy=3)
                   
                   """ ADD IN THE NEW SEG??? or just let it overlap??? """                         
                   #segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = cleaned_seg                        
                   #segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = cleaned_seg + segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
         
                   
                   
                   #print('inference on sublock: ')
                   #print([x, y, z])
                   
                   
     """ Analyze whatever is left """
     if len(batch_x) > 0:
        print('leftover: ' + str(len(batch_x)))
        batch_x = np.vstack(batch_x)
        batch_coords = np.vstack(batch_coords)
        batch_y = np.zeros([batch_size, num_truth_class, quad_depth, quad_size, quad_size])
        
        """ Call transfer to GPU """
        inputs, labels = transfer_to_GPU(batch_x, batch_y, device, mean_arr, std_arr)                                                 
        output_val = unet(inputs)

        """ Convert back to cpu """                                      
        output_tile = output_val.cpu().data.numpy()            
        output_tile = np.moveaxis(output_tile, 1, -1)
        #seg_train = np.argmax(output_tile[0], axis=-1)  
        
        seg_train = np.argmax(output_tile, axis=-1)
         

        ### add in by batch
        for id_b, cleaned_seg in enumerate(seg_train):
            coords = batch_coords[id_b]
            z = coords[0]; x = coords[1]; y = coords[2]
            segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size] = cleaned_seg + segmentation[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
  
                    
     return segmentation