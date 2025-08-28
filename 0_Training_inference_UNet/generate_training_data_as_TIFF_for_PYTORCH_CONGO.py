# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:46:29 2020

@author: tiger
"""

from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:25:37 2018

@author: Neuroimmunology Unit
"""

# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL numexpr!!!
 
@author: Tiger


"""

import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
from skimage import measure
import pickle as pickle
import os
import zipfile
import scipy
# import cv2 as cv
from natsort import natsort_keygen, ns

from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
import glob, os
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import tkinter
from tkinter import filedialog
import os
    
import tifffile as tiff
truth = 0

def plot_max(im, ax=0):
     max_im = np.amax(im, axis=ax)
     plt.figure(); plt.imshow(max_im[:, :])
     return max_im
     

""" removes detections on the very edges of the image """
def clean_edges(im, depth, w, h, extra_z=1, extra_xy=5):
     labelled = measure.label(im)
     cc_coloc = measure.regionprops(labelled)
    
     cleaned_im = np.zeros(np.shape(im))
     for obj in cc_coloc:
         #max_val = obj['max_intensity']
         coords = obj['coords']
         
         bool_edge = 0
         for c in coords:
              if (c[0] <= 0 + extra_z or c[0] >= depth - extra_z):
                   #print('badz')
                   bool_edge = 1
                   break;
              if (c[1] <= 0 + extra_xy or c[1] >= w - extra_xy):
                   #print('badx')
                   bool_edge = 1
                   break;                                       
              if (c[2] <= 0 + extra_xy or c[2] >= h - extra_xy):
                   #print('bady')
                   bool_edge = 1
                   break;                                        
                   
                   
    
         if not bool_edge:
              #print('good')
              for obj_idx in range(len(coords)):
                   cleaned_im[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1

     return cleaned_im                     
            


resize_bool = 0

input_size = 128
depth = 16   # ***OR can be 160
num_truth_class = 1 + 1 # for reconstruction
multiclass = 0



# input_size = 256
# depth = 64   # ***OR can be 160
# num_truth_class = 1 + 1 # for reconstruction
# multiclass = 0

# tf_size = input_size


""" Select multiple folders for analysis AND creates new subfolder for results output """
root = tkinter.Tk()
# get input folders
another_folder = 'y';
list_folder = []
# input_path = "./"
# while(another_folder == 'y'):
#     input_path = filedialog.askdirectory(parent=root, initialdir= input_path,
#                                         title='Please select input directory')
#     input_path = input_path + '/'
    
#     #another_folder = input();   # currently hangs forever
#     another_folder = 'n';

#     list_folder.append(input_path)


list_folder = ['/media/user/4TB_SSD/Training_UNet_Congo/']

        

""" Loop through all the folders and do the analysis!!!"""
for input_path in list_folder:
    foldername = input_path.split('/')[-1]
    sav_dir = input_path + '/' + foldername + '_blocks_128_16_UNet'
 
    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*train_CONGO_im.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('train_CONGO_im.tif','train_TRUTH_IM.tif'), 
                     autofluor=i.replace('train_CONGO_im.tif','train_red_im.tif')) for i in images]

    # zzz
    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("Directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    
    # Required to initialize all
    batch_size = 1;
    
    input_batch = []; truth_batch = [];
    weights = [];
    
    plot_jaccard = [];
    
    output_stack = [];
    output_stack_masked = [];
    all_PPV = [];
    input_im_stack = [];
    
    empty = 1

    """ Do bcolz """
    #import bcolz
    #c = bcolz.carray(a, rootdir = 'test_dir')


    total_samples = 0
    
    expectedLen = 10000
    global_overlap_percent = 0.2
    

    num_files = 0
    for ex in examples:
            #if total_samples > 500:
            #    break
            
            input_name = ex['input']
            input_im = tiff.imread(input_name)
            ### check if there is a problem in the scaling of pixels
            if np.max(input_im) > 30000:
                print('Input not scaled correctly, will rescale')
                break;
                zzz                
              
                
            input_im = np.asarray(input_im, dtype=np.uint16)
            
            
            # truth_im = input_im[:, 1, :, :]
            # input_im = input_im[:, 0, :, :]
            
            
            
            ### check seg_im
            seg_name = ex['truth']
            seg_im = tiff.imread(seg_name)

            
            ### Parse seg_im --- 1 and 3 are segmentations, 2 is to delete
            
            seg_im[seg_im == 2] = 0
            seg_im[seg_im > 3] = 0
            
            seg_im[seg_im > 0] = 1







            ### for debugging, plot truth_im as labelled array using skimage
            #truth_lab = measure.label(truth_im, connectivity=1)
            ### HACK --- DONT DO CRAZY OVERLAP FOR EMPTY TRAINING DATA
            if len(np.unique(seg_im)) == 1:   ### should be > 1 if add in the old uncleaned data
                print('blank')
                overlap_percent = 0
                
            else:
                overlap_percent = global_overlap_percent



            # ### check if there is a problem in the scaling of pixels
            # if np.max(input_im) > 30000:
            #     print('Input STILL not scaled correctly')
 
            
            # truth_name = ex['truth']
            # truth_im = tiff.imread(truth_name)
            # truth_im = np.asarray(truth_im, dtype=np.uint16)   ### sometimes the input dtype is ('<u2') weirdly
                     
            truth_im = seg_im
            
            
   
            """ Analyze each block with offset in all directions """
            quad_size = input_size
            quad_depth = depth
            im_size = np.shape(input_im);
            width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
              
            num_quads = np.floor(width/quad_size) * np.floor(width/quad_size) * np.floor (depth_im/quad_depth);
             

            quad_idx = 1;

          
            segmentation = np.zeros([depth_im, width, height])
            input_im_check = np.zeros(np.shape(input_im))
            total_blocks = 0;


            
            all_xyz = []
            # for x in range(0, width + quad_size, round(quad_size - quad_size * overlap_percent)):
            #      if x + quad_size > width:
            #           difference = (x + quad_size) - width
            #           x = x - difference
                           
                      
            #      # if total_samples >= 5000:
            #      #      #file.close()
            #      #      break   
                      
            #      for y in range(0, height + quad_size, round(quad_size - quad_size * overlap_percent)):
                      
            #           if y + quad_size > height:
            #                difference = (y + quad_size) - height
            #                y = y - difference

            #           for z in range(0, depth_im + quad_depth, round(quad_depth - quad_depth * overlap_percent)):
            #               batch_x = []; batch_y = [];
                          
            #               if z + quad_depth > depth_im:
            #                    difference = (z + quad_depth) - depth_im
            #                    z = z - difference
                          
            
            for z in range(0, depth_im + quad_depth, round(quad_depth - quad_depth * overlap_percent)):
                #batch_x = []; batch_y = [];
      
                if z + quad_depth > depth_im:
                     print('reached end of dim')
                     continue
                     # difference = (z + quad_depth) - depth_im
                     # z = z - difference

                for x in range(0, width + quad_size, round(quad_size - quad_size * overlap_percent)):
                    if x + quad_size > width:
                           print('reached end of dim')
                           continue
                           # difference = (x + quad_size) - width
                           # x = x - difference
                                
                    for y in range(0, height + quad_size, round(quad_size - quad_size * overlap_percent)):
                       
            
                          if y + quad_size > height:
                                print('reached end of dim')
                                continue
                                # difference = (y + quad_size) - height
                                # y = y - difference
                                
                          num_files += 1
                            
                            
                            
                          quad_intensity = input_im[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
                          quad_truth = truth_im[z:z + quad_depth, x:x + quad_size, y:y + quad_size]
                          #quad_truth[quad_truth > 0] = 1
                          
                          """ Clean segmentation by removing objects on the edge """
                          #cleaned_seg = clean_edges(seg_train[0], quad_depth, w=quad_size, h=quad_size, extra_z=1, extra_xy=3)
                          #cleaned_seg = seg_train
                          

                          """ Save block """                          
                          #filename = input_name.split('\\')[-1]  # on Windows
                          filename = input_name.split('/')[-1] # on Ubuntu
                          filename = filename.split('.')[0:-1]
                          filename = '.'.join(filename)
                                                    
                          filename = filename.split('RAW_REGISTERED')[0]


                          """ Check if repeated """
                          skip = 0
                          for coord in all_xyz:
                               if coord == [x,y,z]:
                                    skip = 1
                                    break                      
                               
                          if skip:
                               continue
                               
                          all_xyz.append([x, y, z])  
                           
``````````                          
                          """ If want to save images as well """
                          
          
                          # quad_truth[quad_truth > 0] = 255
                         
                         
                          #max_quad_intensity = plot_max(quad_intensity, ax=0, fig_num=1)
                          #max_quad_truth = plot_max(quad_truth, ax=0, fig_num=2)


                          tiff.imsave(sav_dir + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_block_INPUT.tif', np.uint16(quad_intensity))
                          tiff.imsave(sav_dir + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_block_TRUTH.tif', np.uint16(quad_truth))
                          #imsave(sav_dir + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_quad_INPUT_max_proj.tif', np.uint8(max_quad_intensity))
                          #imsave(sav_dir + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_quad_TRUTH_max_proj.tif', np.uint8(max_quad_truth))                          


                          """ do bcolz """
                          #if empty:                               
                          #     a = bcolz.carray(quad_intensity, expectedlen = expectedLen, rootdir = sav_dir + 'input_im')
                          #     print(a.chunklen)
                          #     b = bcolz.carray(quad_truth, expectedlen = expectedLen,  rootdir = sav_dir + 'truth_im')
                          #     empty = 0
                          #else:
                          #     a.append(quad_intensity)
                          #     a.flush()

                          #     b.append(quad_truth)
                          #     b.flush()
                              
                          total_samples += 1
                          print(total_samples)
  


""" Find mean and std """
images = glob.glob(os.path.join(sav_dir,'*_INPUT.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
examples = [dict(input=i,truth=i.replace('_INPUT.tif','_TRUTH.tif')) for i in images]

sav_dir = sav_dir + '/normalize/'    
try:
    # Create target Directory
    os.mkdir(sav_dir)
    print("Directory " , sav_dir ,  " Created ") 
except FileExistsError:
    print("Directory " , sav_dir ,  " already exists")


""" Calculate mean and std """
import time
start = time.perf_counter()


input_name = examples[0]['input']
input_im = tiff.imread(input_name)

sum_all = np.zeros(np.shape(input_im))
#sum_all = np.expand_dims(sum_all, -1)
save_all = []

""" Calculate mean """
for i in range(len(examples)):              
        """ Load input image """
        input_name = examples[i]['input']
        input_im = tiff.imread(input_name)
        sum_all = sum_all + input_im
mean_im = sum_all/len(examples)
mean_val = np.sum(sum_all)/(mean_im.size * len(examples))

stop = time.perf_counter()

diff = stop - start
print(diff)
""" Calculate standard deviation """
sum_squared_all = np.zeros(np.shape(examples[0]['input']))
for i in range(len(examples)):          
        """ Load input image """
        input_name = examples[i]['input']
        input_im = tiff.imread(input_name)
        sub_mean = input_im - mean_val
        squared = np.square(sub_mean)
        
        sum_squared_all = sum_squared_all + squared

total_sum = np.sum(sum_squared_all)

mean_squared = total_sum/(mean_im.size * len(examples))
square_root = np.sqrt(mean_squared)
std_val = square_root
stop = time.perf_counter()

diff = stop - start
print(diff)

""" Saving the objects """
np.save(sav_dir + 'mean_VERIFIED.npy', mean_val)
np.save(sav_dir + 'std_VERIFIED.npy', std_val)


print('mean_val: ' + str(mean_val))
print('std_val: ' + str(std_val))









