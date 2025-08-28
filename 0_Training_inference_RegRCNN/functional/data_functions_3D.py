# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 14:20:44 2019

@author: Tiger
"""

import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from skimage import measure
from natsort import natsort_keygen, ns
import os
import pickle
import scipy.io as sio
import math
#import zipfile
#import bz2

#from plot_functions import *
#from data_functions import *
#from post_process_functions import *
#from UNet import *

import glob, os

#from off_shoot_functions import *


from PIL import ImageSequence
def open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default'):
    input_im = [];
    tmp = Image.open(input_name)
    depth_of_im = 0
    for L, page in enumerate(ImageSequence.Iterator(tmp)):
       page = np.asarray(page, np.float32)
       input_im.append(page)
       depth_of_im = depth_of_im + 1

    input_im = np.asarray(input_im, dtype=np.float32)
    
    """ also detect shape of input_im and adapt accordingly """
    width_im = np.shape(input_im)[1]
    height_im = np.shape(input_im)[2]
    
    """ Skips this next cropping if we want just the default input size """
    if width_max != 'default' or height_max != 'default' or depth !='default':
            """ If image has same or larger number of stacks than requested, take all stacks up to requested height """
            if depth_of_im > depth:
                mid_depth_of_im = int(depth_of_im/2)
                mid_depth = int(depth/2)
                
                start_depth = mid_depth_of_im - mid_depth
                end_depth = mid_depth_of_im + mid_depth
                
                input_im = input_im[start_depth:end_depth, :, :]
            else:   # if smaller
                tmp = np.zeros([depth, width_im, height_im])       # if stack is smaller than requested, add additional blank stacks
                tmp[0:depth_of_im, :, :] = input_im
                input_im = tmp
            
            """ Also make sure that the height/width are okay """
            if width_im >= width_max:
                mid_width_of_im = int(width_im/2)
                mid_width = int(width_max/2)
                
                start_width = mid_width_of_im - mid_width
                end_width = mid_width_of_im + mid_width            
            
                input_im = input_im[:, start_width:end_width, :]
            else:  # if smaller
                tmp = np.zeros([depth, width_max, height_im])       # if stack is smaller than requested, add additional blank stacks
                tmp[:, 0:width_im, :] = input_im
                input_im = tmp
                

            """ Also make sure that the height/width are okay """
            if height_im >= height_max:
            
                mid_height_of_im = int(height_im/2)
                mid_height = int(height_max/2)
                
                start_height = mid_height_of_im - mid_height
                end_height = mid_height_of_im + mid_height
            
                input_im = input_im[:, :, start_height:end_height]
            else:  # if smaller
                tmp = np.zeros([depth, width_max, height_max])       # if stack is smaller than requested, add additional blank stacks
                tmp[:, :, 0:height_im] = input_im
                input_im = tmp
                

    return input_im





""" Loads single channel truth data """
def load_truth_3D(truth_name, width_max, height_max, depth, spatial_weight_bool=0, pick=0, delete_seed=0):
    if not pick:
         truth_tmp = [];
         tmp = Image.open(truth_name)
         depth_of_im = 0
         for L, page in enumerate(ImageSequence.Iterator(tmp)):
            page = np.asarray(page, np.float32)
            truth_tmp.append(page)
            depth_of_im = depth_of_im + 1
     
         truth_tmp = np.asarray(truth_tmp, dtype=np.float32)
    else:
        with open(truth_name, 'rb') as f:  # Python 3: open(..., 'rb')
           #print('HEY' + truth_name)
           loaded = pickle.load(f)
           truth_tmp = loaded[0]
           depth_of_im = np.shape(truth_tmp)
           depth_of_im = depth_of_im[0]
           
       
    """ also detect shape of input_im and adapt accordingly """
    width_im = np.shape(truth_tmp)[1]
    height_im = np.shape(truth_tmp)[2]

    #truth_tmp = truth_tmp[:, 0:input_size, 0:input_size]

    """ If image has same or larger number of stacks than requested, take all stacks up to requested height """
    if depth_of_im > depth:
        mid_depth_of_im = int(depth_of_im/2)
        mid_depth = int(depth/2)
                
        start_depth = mid_depth_of_im - mid_depth
        end_depth = mid_depth_of_im + mid_depth
        truth_tmp = truth_tmp[start_depth:end_depth, :, :]
    else:   # if smaller
        tmp = np.zeros([depth, width_im, height_im])       # if stack is smaller than requested, add additional blank stacks
        tmp[0:depth_of_im, :, :] = truth_tmp
        truth_tmp = tmp
    
    """ Also make sure that the height/width are okay """
    if width_im >= width_max:
        mid_width_of_im = int(width_im/2)
        mid_width = int(width_max/2)
              
        start_width = mid_width_of_im - mid_width
        end_width = mid_width_of_im + mid_width   
    
        truth_tmp = truth_tmp[:, start_width:end_width, :]
    else:  # if smaller
        tmp = np.zeros([depth, width_max, height_im])       # if stack is smaller than requested, add additional blank stacks
        tmp[:, 0:width_im, :] = truth_tmp
        truth_tmp = tmp       

    """ Also make sure that the height/width are okay """
    if height_im >= height_max:
            
        mid_height_of_im = int(height_im/2)
        mid_height = int(height_max/2)
                
        start_height = mid_height_of_im - mid_height
        end_height = mid_height_of_im + mid_height
    
    
        truth_tmp = truth_tmp[:, :, start_height:end_height]
    else:  # if smaller
        tmp = np.zeros([depth, width_max, height_max])       # if stack is smaller than requested, add additional blank stacks
        tmp[:, :, 0:height_im] = truth_tmp
        truth_tmp = tmp
     
    """ if pickle of spatial weights, return here """
    if pick:
         truth_im = truth_tmp
         return truth_im, np.zeros(np.shape(truth_im))
    
    """ convert truth to 2 channel image """
    channel_1 = np.copy(truth_tmp)
    channel_1[channel_1 == 0] = -1
    channel_1[channel_1 > 0] = 0
    channel_1[channel_1 == -1] = 1
            
    channel_2 = np.copy(truth_tmp)
    channel_2[channel_2 > 0] = 1   
    
    truth_im = np.zeros(np.shape(truth_tmp) + (2,))
    
    if not delete_seed.any():
        truth_im[:, :, :, 0] = channel_1   # background
        truth_im[:, :, :, 1] = channel_2   # blebs
        
    else:
        channel_1[delete_seed > 0] = 1
        truth_im[:, :, :, 0] = channel_1   # background
        channel_2[delete_seed > 0] = 0
        truth_im[:, :, :, 1] = channel_2   # blebs
        
        
    blebs_label = np.copy(truth_im[:, :, :, 1])   # ONLY WEIGHTS THE NON-background!!!
    
    if spatial_weight_bool:
        """ Get spatial AND class weighting mask for truth_im """
        sp_weighted_labels = spatial_weight(blebs_label,edgeFalloff=10,background=0.01,approximate=True)
        
        #numpy.unique(sp_weighted_labels)         # TO DEBUG AND CHECK THAT THE GRADIENT IS ACTUALLY APPLIED!

        """ OR DO class weighting ONLY """
        #c_weighted_labels = class_weight(blebs_label, loss, weight=10.0)        
        
        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_im)
        weighted_labels[:, :, :, 1] = sp_weighted_labels
    else:
        weighted_labels = np.zeros(np.shape(truth_im))
        
    return truth_im, weighted_labels

              
""" Loads in multi-class images """ 
def load_class_truth_3D(truth_name, num_truth_class, width_max, height_max, depth, spatial_weight_bool=1, splitter='truth', pick=0, class_num=0, delete_seed=0, skip_class=-1, resized_check=0):
    split = truth_name.split('.')   # takes away the ".tif" filename at the end
    truth_name = split[0:-1]
    truth_name = '.'.join(truth_name)
                    
    split = truth_name.split('\\')   # takes away the "path // stuff" 
    truth_name = split[-1]
    input_path = split[0]

    split = truth_name.split(splitter)   # takes awayeverything after the "truth" 
    truth_name = split[0]
    truth_name = truth_name + splitter
    
    classes = glob.glob(os.path.join(input_path, truth_name + '*.*'))
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    classes.sort(key = natsort_key1)

    if num_truth_class <= 2:
        truth_name = classes[class_num]
        truth_im, weighted_labels = load_truth_3D(truth_name, width_max, height_max, depth, spatial_weight_bool=spatial_weight_bool, pick=pick, delete_seed=delete_seed)
        
        """ For pickle must also generate background channel """
        if pick:
             truth_im, weighted_labels = load_truth_3D(truth_name, width_max, height_max, depth, spatial_weight_bool=spatial_weight_bool, pick=pick, delete_seed=delete_seed)

             all_weighted = np.zeros([depth, width_max, height_max, num_truth_class]);
             all_weighted[:, :, :, 1] = truth_im
             truth_im = all_weighted
        
    elif num_truth_class > 2 and pick:  
        """ if is a pickle file containing spatial weights """
        all_classes = np.zeros([depth, width_max, height_max, num_truth_class]); 
        all_weighted = np.zeros([depth, width_max, height_max, num_truth_class]);
        class_idx = 1;
        for truth_name in classes:
            if class_idx == num_truth_class:
                break
            truth_class, weighted_labels_class = load_truth_3D(truth_name, width_max, height_max, depth, spatial_weight_bool=0, pick=pick, delete_seed=delete_seed)
            all_classes[:, :, :, class_idx] = truth_class[:, :, :]
            all_weighted[:, :, :, class_idx] = weighted_labels_class[:, :, :]
            class_idx += 1         
        truth_im = np.copy(all_classes)
        weighted_labels = np.copy(all_weighted)            
            
            
            
    else:
        all_classes = np.zeros([depth, width_max, height_max, num_truth_class]); 
        all_weighted = np.zeros([depth, width_max, height_max, num_truth_class]);
        class_idx = 1;
        loop_idx = 1;
        for truth_name in classes:
        
            #print(truth_name)
            #file = open('testfile.txt','w') 
            #file.write('Hello World')  

        
            #with open('output.txt', 'a+') as file:  # Use file to refer to the file object
            #    file.write('Hi there!')
            #    file.write(truth_name)
            #    file.close() 
            if skip_class == class_idx:
                 class_idx += 1
                 continue;
              
            if loop_idx == num_truth_class:
                break
            truth_class, weighted_labels_class = load_truth_3D(truth_name, width_max, height_max, depth, spatial_weight_bool=0, delete_seed=delete_seed)
            all_classes[:, :, :, loop_idx] = truth_class[:, :, :, 1]
            all_weighted[:, :, :, loop_idx] = weighted_labels_class[:, :, :, 1]
            class_idx += 1
            loop_idx += 1
            
        
        
        #""" Eliminate all 2nd class weights/labels """
        #all_classes[:, :, :, 2] = np.zeros([depth, input_size, input_size])
        #all_weighted[:, :, :, 2] = np.zeros([depth, input_size, input_size])
              
        truth_im = np.copy(all_classes)
        weighted_labels = np.copy(all_weighted)
        
        
           
        """ If resized, check to make sure no straggling non-attached objects """
        if resized_check:
             truth_im = check_resized(truth_im, depth, width_max, height_max)
             
        
        """ Ensure full semantic segmentation ==> no overlap between any channels 
            mostly just subtract channel1 - (channel 2 + channel 3), channel 2 - channel 3
        """
        for channel_idx in range(len(truth_im[0, 0, 0, :])):
             #ch1 = np.copy(truth_im[:, :, :, 1])
             #ch2 = np.copy(truth_im[:, :, :, 2])
             ch_orig = np.copy(truth_im[:, :, :, channel_idx])
             
             ch_to_subtract = np.zeros(np.shape(ch_orig))
             loop_idx = channel_idx + 1
             while loop_idx < len(truth_im[0, 0, 0, :]):
                  ch_to_subtract = np.add(ch_to_subtract, truth_im[:, :, :, loop_idx])
                  loop_idx += 1
             ch_to_subtract[ch_to_subtract > 0] = 1  # binarize
             
             no_overlap = np.subtract(ch_orig, ch_to_subtract)
             no_overlap[no_overlap == -1] = 0
             
             truth_im[:, :, :, channel_idx] = no_overlap
        
        #add = np.add(ch1, no_overlap_ch2)
        #ma = np.amax(add, axis=0)
        #plt.figure(); plt.imshow(ma)        
                
        """ Ensure full semantic segmentation ==> no overlap between any channels """
#        ch1_max = np.copy(weighted_labels[:, :, :, 1])
#        ch2_max = np.copy(weighted_labels[:, :, :, 2])
#        
#        no_overlap_ch1 = np.subtract(ch1_max, ch2_max)
#        no_overlap_ch1[no_overlap_ch1 == -1] = 0
#        
#        add = np.add(ch1_max, ch2_max)
#        ma = np.amax(add, axis=0)
#        plt.figure(); plt.imshow(ma)
#
#        ma_ch2 = np.amax(ch1_max, axis=0)
#        plt.figure(); plt.imshow(ma_ch2)

        """ Apply spatial weight here instead to ensure semantic segmentation correct """
        if spatial_weight_bool:
             for w_idx in range(num_truth_class - 1):
                  channel_truth = truth_im[:, :, :, w_idx + 1]
                  """ Get spatial AND class weighting mask for truth_im """
                  sp_weighted_labels = spatial_weight(channel_truth,edgeFalloff=10,background=0.01,approximate=True)
                                  
                  """ Create a matrix of weighted labels """
                  weighted_labels[:, :, :, w_idx + 1] = sp_weighted_labels
                  #print(w_idx)


        """ Create un-weighted background """        
        background = np.zeros([depth, width_max, height_max])
        for idx in range(num_truth_class - 1):
            background = background + truth_im[:, :, :, idx + 1]
            #print(idx + 1)
        background[background > 0] = -1
        background[background == 0] = 1
        background[background == -1] = 0
        #background = np.expand_dims(background, axis=-1)
                
        truth_im[:, :, :, 0] = background
        weighted_labels[:, :, :, 0] = background
       
    return truth_im, weighted_labels



""" Loads in multi-class images """ 
def load_class_truth_3D_single_class(truth_name, num_truth_class, input_size, depth, spatial_weight_bool=1):
    split = truth_name.split('.')   # takes away the ".tif" filename at the end
    truth_name = split[0:-1]
    truth_name = '.'.join(truth_name)
                    
    split = truth_name.split('\\')   # takes away the "path // stuff" 
    truth_name = split[-1]
    input_path = split[0]

    split = truth_name.split('truth')   # takes awayeverything after the "truth" 
    truth_name = split[0]
    truth_name = truth_name + 'truth'

    classes = glob.glob(os.path.join(input_path, truth_name + '*.tif*'))
     
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    classes.sort(key = natsort_key1)
    
    if num_truth_class <= 2:
        truth_name = classes[0]
        truth_im, weighted_labels = load_truth_3D(truth_name, input_size, depth, spatial_weight_bool=1)
    else:
        all_classes = np.zeros([depth, input_size, input_size, num_truth_class]); 
        all_weighted = np.zeros([depth, input_size, input_size, num_truth_class]);
        class_idx = 1;
        for truth_name in classes:
            print(truth_name)
            if class_idx == num_truth_class:
                break
            truth_class, weighted_labels_class = load_truth_3D(truth_name, input_size, depth, spatial_weight_bool=1)
            all_classes[:, :, :, class_idx] = truth_class[:, :, :, 1]
            all_weighted[:, :, :, class_idx] = weighted_labels_class[:, :, :, 1]
            class_idx += 1

        """ Eliminate all 2nd class weights/labels """
        all_classes[:, :, :, 2] = np.zeros([depth, input_size, input_size])
        all_weighted[:, :, :, 2] = np.zeros([depth, input_size, input_size])
              
        truth_im = np.copy(all_classes)
        weighted_labels = np.copy(all_weighted)
        
        """ Create un-weighted background """        
        background = np.zeros([depth, input_size, input_size])
        for idx in range(num_truth_class - 1):
            background = background + truth_im[:, :, :, idx + 1]
            print(idx + 1)
        background[background > 0] = -1
        background[background == 0] = 1
        background[background == -1] = 0
        #background = np.expand_dims(background, axis=-1)
                
        truth_im[:, :, :, 0] = background
        weighted_labels[:, :, :, 0] = background
       
    return truth_im, weighted_labels



""" Convert voxel list to array """
def convert_vox_to_matrix(voxel_idx, zero_matrix):
    for row in voxel_idx:
        #print(row)
        zero_matrix[(row[0], row[1], row[2])] = 1
    return zero_matrix


""" For plotting the output as an interactive scroller"""
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


""" Only keeps objects in stack that are 5 slices thick!!!"""
def slice_thresh(output_stack, slice_size=5):
    binary_overlap = output_stack > 0
    labelled = measure.label(binary_overlap)
    cc_overlap = measure.regionprops(labelled)
    
    all_voxels = []
    all_voxels_kept = []; total_blebs_kept = 0
    all_voxels_elim = []; total_blebs_elim = 0
    total_blebs_counted = len(cc_overlap)
    for bleb in cc_overlap:
        cur_bleb_coords = bleb['coords']
    
        # get only z-axis dimensions
        z_axis_span = cur_bleb_coords[:, -1]
    
        min_slice = min(z_axis_span)
        max_slice = max(z_axis_span)
        span = max_slice - min_slice
    
        """ ONLY KEEP OBJECTS that span > 5 slices """
        if span >= slice_size:
            print("WIDE ENOUGH object") 
            if len(all_voxels_kept) == 0:   # if it's empty, initialize
                all_voxels_kept = cur_bleb_coords
            else:
                all_voxels_kept = np.append(all_voxels_kept, cur_bleb_coords, axis = 0)
                
            total_blebs_kept = total_blebs_kept + 1
        else:
            print("NOT wide enough")
            if len(all_voxels_elim) == 0:
                print("came here")
                all_voxels_elim = cur_bleb_coords
            else:
                all_voxels_elim = np.append(all_voxels_elim, cur_bleb_coords, axis = 0)
                
            total_blebs_elim = total_blebs_elim + 1
       
        if len(all_voxels) == 0:   # if it's empty, initialize
            all_voxels = cur_bleb_coords
        else:
            all_voxels = np.append(all_voxels, cur_bleb_coords, axis = 0)
            
    print("Total kept: " + str(total_blebs_kept) + " Total eliminated: " + str(total_blebs_elim))
    
    
    """ convert voxels to matrix """
    all_seg = convert_vox_to_matrix(all_voxels, np.zeros(output_stack.shape))
    all_blebs = convert_vox_to_matrix(all_voxels_kept, np.zeros(output_stack.shape))
    all_eliminated = convert_vox_to_matrix(all_voxels_elim, np.zeros(output_stack.shape))
    
    return all_seg, all_blebs, all_eliminated


""" Find vectors of movement and eliminate blobs that migrate """
def distance_thresh(all_blebs_THRESH, average_thresh=15, max_thresh=15):
    
    # (1) Find and plot centroid of each 2D image object:
    centroid_matrix_3D = np.zeros(np.shape(all_blebs_THRESH))
    for i in range(len(all_blebs_THRESH[0, 0, :])):
        bin_cur_slice = all_blebs_THRESH[:, :, i] > 0
        label_cur_slice = measure.label(bin_cur_slice)
        cc_overlap_cur = measure.regionprops(label_cur_slice)
        
        for obj in cc_overlap_cur:
            centroid_matrix_3D[(int(obj['centroid'][0]),) + (int(obj['centroid'][1]),) + (i,)] = 1   # the "i" puts the centroid in the correct slice!!!
        
        #print(i)
        
    # (2) use 3D cc_overlap to find clusters of centroids
    binary_overlap = all_blebs_THRESH > 0
    labelled = measure.label(binary_overlap)
    cc_overlap_3D = measure.regionprops(labelled)
        
    all_voxels_kept = []; num_kept = 0
    all_voxels_elim = []; num_elim = 0
    for obj3D in cc_overlap_3D:
        
        slice_idx = np.unique(obj3D['coords'][:, -1])
        
        cropped_centroid_matrix = centroid_matrix_3D[:, :, min(slice_idx) : max(slice_idx) + 1]
        
        mask = np.ones(np.shape(cropped_centroid_matrix))

        translate_z_coords = obj3D['coords'][:, 0:2]
        z_coords = obj3D['coords'][:, 2:3]  % min(slice_idx)   # TRANSLATES z-coords to 0 by taking modulo of smallest slice index!!!
        translate_z_coords = np.append(translate_z_coords, z_coords, -1)
        
        obj_mask = convert_vox_to_matrix(translate_z_coords, np.zeros(cropped_centroid_matrix.shape))
        mask[obj_mask == 1] = 0 

        tmp_centroids = np.copy(cropped_centroid_matrix)  # contains only centroids that are masked by array above
        tmp_centroids[mask == 1] = 0
        
        
        ##mask = np.ones(np.shape(centroid_matrix_3D))
        ##obj_mask = convert_vox_to_matrix(obj3D['coords'], np.zeros(output_stack.shape))
        ##mask[obj_mask == 1] = 0 
    
        ##tmp_centroids = np.copy(centroid_matrix_3D)  # contains only centroids that are masked by array above
        ##tmp_centroids[mask == 1] = 0
        
        cc_overlap_cur_cent = measure.regionprops(np.asarray(tmp_centroids, dtype=np.int))  
        
        list_centroids = []
        for centroid in cc_overlap_cur_cent:
            if len(list_centroids) == 0:
                list_centroids = centroid['coords']
            else:
                list_centroids = np.append(list_centroids, centroid['coords'], axis = 0)
    
        sorted_centroids = sorted(list_centroids,key=lambda x: x[2])  # sort by column 3
        
        
        """ Any object with only 1 or less centroids is considered BAD, and is eliminated"""
        if len(sorted_centroids) <= 1:
            num_elim = num_elim + 1
            
            if len(all_voxels_elim) == 0:   # if it's empty, initialize
                all_voxels_elim = obj3D['coords']
            else:
                all_voxels_elim = np.append(all_voxels_elim, obj3D['coords'], axis = 0)
            continue;
        
    
        # (3) Find distance from 1st - 2nd - 3rd - 4th - 5th ect... centroids
        all_distances = []
        for i in range(len(sorted_centroids) - 1):
            center_1 = sorted_centroids[i]
            center_2 = sorted_centroids[i + 1]
            
            # Find distance:
            dist = math.sqrt(sum((center_1 - center_2)**2))           # DISTANCE FORMULA
            #print(dist)
            all_distances.append(dist)
        average_dist = sum(all_distances)/len(all_distances)
        max_dist = max(all_distances)
        
        
        # (4) If average distance is BELOW thresdhold, then keep the 3D cell body!!!
        # OR, if max distance moved > 15 pixels
        #print("average dist is: " + str(average_dist))
        if average_dist < average_thresh or max_dist < max_thresh:
            if len(all_voxels_kept) == 0:   # if it's empty, initialize
                all_voxels_kept = obj3D['coords']
            else:
                all_voxels_kept = np.append(all_voxels_kept, obj3D['coords'], axis = 0)
            
            num_kept = num_kept + 1
        else:
            num_elim = num_elim + 1
            
            if len(all_voxels_elim) == 0:   # if it's empty, initialize
                all_voxels_elim = obj3D['coords']
            else:
                all_voxels_elim = np.append(all_voxels_elim, obj3D['coords'], axis = 0)
            
        print("Finished distance thresholding for: " + str(num_elim + num_kept) + " out of " + str(len(cc_overlap_3D)) + " images")
    
    final_bleb_matrix = convert_vox_to_matrix(all_voxels_kept, np.zeros(all_blebs_THRESH.shape))
    elim_matrix = convert_vox_to_matrix(all_voxels_elim, np.zeros(all_blebs_THRESH.shape))
    print('Kept: ' + str(num_kept) + " eliminated: " + str(num_elim))
    
    return final_bleb_matrix, elim_matrix




""" converts a matrix into a multipage tiff (DEPTH FIRST) to save!!! """
def convert_matrix_to_multipage_tiff(matrix):
    rolled = np.rollaxis(matrix, axis=2, start=0).shape  # changes axis to be correct sizes
    tiff_image = np.zeros(rolled)
    for i in range(len(tiff_image)):
        tiff_image[i, :, :] = matrix[:, :, i]
    return tiff_image


""" mutlipage tiff to a matrix (DEPTH LAST) """
def convert_multitiff_to_matrix(tiff_image):
    rolled = np.rollaxis(tiff_image, axis=0, start=3).shape  # changes axis to be correct sizes
    matrix = np.zeros(rolled)
    for i in range(len(matrix[0, 0, :])):
        matrix[:, :, i] = tiff_image[i, :, :]
    return matrix

