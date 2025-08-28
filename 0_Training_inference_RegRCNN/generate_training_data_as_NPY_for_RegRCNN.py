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
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
from skimage import measure
import pickle as pickle
import os
import scipy
from natsort import natsort_keygen, ns

#from plot_functions_CLEANED import *
#from data_functions_CLEANED import *
#from data_functions_3D import *
import glob, os
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import tkinter
from tkinter import filedialog
import os
    
import tifffile as tiff

                          
#import nibabel as nib
import json

import pandas as pd
import random as random            

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
            



# import matlab.engine
# print('Running watershed in MATLAB')
# eng = matlab.engine.start_matlab()
# s = eng.genpath('./MATLAB_functions/')
# eng.addpath(s, nargout=0)


resize_bool = 0

input_size = 128
depth = 16   # ***OR can be 160


# input_size = 64
# depth = 16   # ***OR can be 160

num_truth_class = 1 + 1 # for reconstruction
multiclass = 0

connectivity = 1

""" FOR WATERSHED SEGMENTATION """
water = 0



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


list_folder = ['/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/']





        

""" Loop through all the folders and do the analysis!!!"""
for input_path in list_folder:
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_RegRCNN_CHECKING'
 
    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_input_im_CROP.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_input_im_CROP.tif','_seg_CROP.tif'), 
                     maskrcnn=i.replace('_input_im_CROP.tif','_cleaned_NEW_CROP.tif'),
                     maskrcnn_8bit=i.replace('_input_im_CROP.tif','_cleaned_NEW_RGB_to_8bit_CROP.tif'),
                     ) for i in images]


    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("Directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("Directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'




    ### make more subfolders for truth, labels, and test set
    sav_dir_imagesTr = sav_dir + '/train/'
    sav_dir_imagesTs = sav_dir + '/test/'
    sav_dir_WATER = sav_dir + '/watershed/'
    #zzz
   
    try:
        os.mkdir(sav_dir_imagesTr)
        os.mkdir(sav_dir_imagesTs)
        os.mkdir(sav_dir_WATER)
        
        
    except FileExistsError:
        print("Directory " , sav_dir_imagesTr ,  " already exists")
            
    
    

    ### use random number generator to split into test set
    random.seed(10)

    
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
    overlap_percent = 0.4  #0.2
    
    
    min_size = 20
    
    info_dict_train = []
    info_dict_test  = []
    all_num_objs = []
    #for i in range(0, len(images), 2):
            #if total_samples > 500:
            #    break
    for i in range(len(examples)):  
        
            
        
            input_name = examples[i]['input']
            input_im = tiff.imread(input_name)
            ### check if there is a problem in the scaling of pixels
            if np.max(input_im) > 30000:
                print('Input not scaled correctly, will rescale')
                
                zzz
                break;
                                
            

                
            
            input_im = np.asarray(input_im, dtype=np.uint16)
            
            ### check if there is a problem in the scaling of pixels
            if np.max(input_im) > 30000:
                print('Input STILL not scaled correctly')
 
            
            
            truth_name = examples[i]['truth']
            truth_im = tiff.imread(truth_name)
            truth_im = np.asarray(truth_im, dtype=np.uint16)   ### sometimes the input dtype is ('<u2') weirdly
           
            
            ### for debugging, plot truth_im as labelled array using skimage
            #truth_lab = measure.label(truth_im, connectivity=1)
            ### HACK --- DONT DO CRAZY OVERLAP FOR EMPTY TRAINING DATA
            if len(np.unique(truth_im)) == 1:   ### should be 5 if add in the old uncleaned data
                print('blank')
                overlap_percent = 0


            

            mask = examples[i]['maskrcnn']
            try:
                mask = tiff.imread(mask)
            except:
                mask = []
            mask = np.asarray(mask, dtype=np.uint16)   ### sometimes the input dtype is ('<u2') weirdly
           
            
            mask_8bit = examples[i]['maskrcnn_8bit']
            try:
                mask_8bit = tiff.imread(mask_8bit)
            except:
                mask_8bit = []
            mask_8bit = np.asarray(mask_8bit, dtype=np.uint16)   ### sometimes the input dtype is ('<u2') weirdly
        
        
        
   
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
 

            padx = 25
            pady = 15
            padz = 4
    
            ### Add blank edges ONLY for the non-maskrcnn corrected original data
            #if len(mask) == 0:
            new_dim_im = np.zeros([depth_im + padz * 2, width + padx * 2, height + pady * 2])
            new_dim_im[padz: padz + depth_im, padx: padx + width, pady: pady + height] = input_im
            input_im = new_dim_im
            
            
            ### Add blank edges to truth im
            new_dim_im = np.zeros([depth_im + padz * 2, width + padx * 2, height + pady * 2])
            new_dim_im[padz: padz + depth_im, padx: padx + width, pady: pady + height] = truth_im
            truth_im = new_dim_im  
            
            
            if len(mask) > 0:
                ### Add blank edges to truth im
                new_dim_im = np.zeros([depth_im + padz * 2, width + padx * 2, height + pady * 2])
                new_dim_im[padz: padz + depth_im, padx: padx + width, pady: pady + height] = mask
                mask = new_dim_im  
                
                
                ### Add blank edges to truth im
                new_dim_im = np.zeros([depth_im + padz * 2, width + padx * 2, height + pady * 2])
                new_dim_im[padz: padz + depth_im, padx: padx + width, pady: pady + height] = mask_8bit
                mask_8bit = new_dim_im  
            
            
            im_size = np.shape(input_im); width = im_size[1];  height = im_size[2]; depth_im = im_size[0];  




            ### CLEAN UP TRUTH:
                
            if len(mask) == 0:
            
            
                truth_im[truth_im < 255] = 0
                truth_im[truth_im > 0] = 255

            else:
                
                ### This is new corrected ground truth data
                
                
                """ Things to fix: 
                    
                        - very small number cells were lost in RGB conversion
                        
                        - missing cells that have IDs that are too similar next to each other - so didnt correct them
                        - green and blue labels too strict - sometimes deleting whole neighboring cells
                        
                        - put full cells back in after subtracting edges
                    """
                
                
                # first subtract out all areas where label == 1  AND where label == 6
                truth_im = np.asarray(truth_im, dtype=int)
                
                mask[truth_im == 1] = 0
                mask[truth_im == 6] = 0
                mask = np.asarray(mask, dtype=int)
                
                # then loop through maskrcnn connected components
                cc = measure.regionprops(mask, intensity_image=mask)
                
                
                cleaned_truth = np.zeros(np.shape(truth_im))
                label4 = np.zeros(np.shape(truth_im))
                
                cell_count = 1
                lab4_count = 1
                for cell_id, cell in enumerate(cc):
                    
                    coords = cell['coords']
                    
                    # any that touch label == 2 or label == 3 --> delete b/c fully new cell
                    if len(np.where(truth_im[coords[:, 0], coords[:, 1], coords[:, 2]] == 2)[0]) > 0 or len(np.where(truth_im[coords[:, 0], coords[:, 1], coords[:, 2]] == 3)[0]) > 0:
                
                        continue
                    
                    if len(coords) < min_size:
                        continue
                    
                    
                    ### if matches value of 4 OR HACK -- greater than 4 as well for the moment, then save as separate cell
                    if len(np.where(truth_im[coords[:, 0], coords[:, 1], coords[:, 2]] == 4)[0]) > 0:
                        
                        label4[coords[:, 0], coords[:, 1], coords[:, 2]] = lab4_count
                        lab4_count += 1
                    
                    else:
                        # for any other unique labels add them to array
                        cleaned_truth[coords[:, 0], coords[:, 1], coords[:, 2]] = cell_count
                        
                        cell_count += 1
                    
                #plot_max(cleaned_truth)
                
                    
                ### add in all the label == 2 as fully new cell and label == 3 as fully new cell
                label2 = np.copy(truth_im)
                label2[label2 != 2] = 0
                label2 = measure.label(label2, connectivity=1)
                cc = measure.regionprops(label2)
                for cell_id, cell in enumerate(cc):
                    
                    coords = cell['coords']
                    cleaned_truth[coords[:, 0], coords[:, 1], coords[:, 2]] = np.max(cleaned_truth) + 1
                    
                #plot_max(cleaned_truth)
                    
                label3 = np.copy(truth_im)
                label3[label3 != 3] = 0
                label3 = measure.label(label3, connectivity=1)
                cc = measure.regionprops(label3)
                for cell_id, cell in enumerate(cc):
                    
                    coords = cell['coords']
                    cleaned_truth[coords[:, 0], coords[:, 1], coords[:, 2]] = np.max(cleaned_truth) + 1                    
                #plot_max(cleaned_truth)
                
                
                ### Add in all label == 6 as new cells
                label6= np.copy(truth_im)
                label6[label6 != 6] = 0
                label6 = measure.label(label6, connectivity=1)
                cc = measure.regionprops(label6)
                for cell_id, cell in enumerate(cc):
                    
                    coords = cell['coords']
                    cleaned_truth[coords[:, 0], coords[:, 1], coords[:, 2]] = np.max(cleaned_truth) + 1                 
                
                
                ### also add in all labels == 4 by combining with all of label == 4 cells HACK > 4 is fine right now
                
                label4_truth = np.copy(truth_im)
                label4_truth[label4_truth < 4] = 0
                label4_truth[label4_truth > 4] = 0
                label4_truth[label4_truth == 4] = 1
                
                lab = measure.label(label4_truth, connectivity=1)
                cc = measure.regionprops(lab)
                 
                 
                 
                #label4[label4_truth == 4] = 1
                #label4 = measure.label(label4)                
                ### Then loop through and add these as well
                label4 = np.asarray(label4, dtype=int)
                arr_match = []
                cc_lab4 = measure.regionprops(label4)
                for cell in cc_lab4:
                    arr_match.append(cell['coords'])
                
                
                
                for obj_id, obj in enumerate(cc):
                    
                    coords = obj['coords']
                    vals = label4[coords[:, 0], coords[:, 1], coords[:, 2]] 
                    
                    # remove all zeros
                    vals = vals[vals !=0]
                    
                    unq, counts = np.unique(vals, return_counts=True)
                    
                    if len(unq) > 1:
                        print('more than one matched, pick highest occurence')
                        
                        unq = unq[np.argmax(counts)]
                        
                    
                    
                    elif len(unq) == 1:
                        unq = unq[0]
                        
                    else:
                        print('no matching found')
                        continue
                        
                        
                    # add coords to list of coordinates
                    add = np.concatenate((arr_match[unq - 1], coords))  ### -1 because starts from 0
                    arr_match[unq - 1] = add

                
                for cell in arr_match:
                    cleaned_truth[cell[:, 0], cell[:, 1], cell[:, 2]] = np.max(cleaned_truth) + 1
                  
                
                
                #plot_max(cleaned_truth)              
                
                ### Also set labels == 7 where any cells connected by a number 7 object are assigned to the same cell ID
                label7= np.copy(truth_im)
                label7[label7 != 7] = 0
                label7 = measure.label(label7, connectivity=1)
                cc = measure.regionprops(label7)
                for cell_id, cell in enumerate(cc):
                    #zzz
                   # print('match')
                    coords = cell['coords']
                    vals = cleaned_truth[coords[:, 0], coords[:, 1], coords[:, 2]]         
                    vals = np.unique(vals)
                    vals = vals[vals != 0]
                    
                    if len(vals) == 0:
                         continue                
                    assign_val = np.max(vals)
                    ### Then loop through each object in cleaned_truth with value of vals and set it to assign_val

                    #if assign_val == 0:
                    #    zzz
                        
                    for val in vals:
                        cleaned_truth[cleaned_truth == val] = assign_val
                        #print(val)

                
                
                truth_im = cleaned_truth
                
                            
                            


            
            #truth_lab = measure.label(truth_im, connectivity=1)
            tiff.imwrite(sav_dir + str(i) + '_truth_lab.tif', np.asarray(cleaned_truth, dtype=np.uint16))
            
            
            #truth_lab = measure.label(input_im, connectivity=1)
            tiff.imwrite(sav_dir + str(i) + '_input_im.tif', np.asarray(input_im, dtype=np.uint16))
                        
            
            im_size = np.shape(input_im);
            width = im_size[1];  height = im_size[2]; depth_im = im_size[0];                 
        
        
        
            all_xyz = [] 
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
                           
                            
                        print([x, y, z])
                        

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
                        
                        

                             
                        all_xyz.append([x, y, z])  
                         
                        
                        """ If want to save images as well """
                         
                        #import napari
                        #viewer = napari.view_image(quad_truth); viewer.add_image(quad_intensity)
                        
                        
                           
                        
                           
                        #max_quad_intensity = plot_max(quad_intensity, ax=0, fig_num=1)
                        #max_quad_truth = plot_max(quad_truth, ax=0, fig_num=2)
                        
                        #zzz
                        #data = np.arange(4*4*3).reshape(4,4,3)
                        quad_intensity_nib = np.moveaxis(quad_intensity, 0, -1)
                          
                        #quad_intensity_nib = nib.Nifti1Image(quad_intensity_nib, affine=np.eye(4))
                        
                        rand_int = random.randint(1, 10)  ### 10% for testing
                        if rand_int == 10:                              
                            #np.save(sav_dir_imagesTs + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_' + str(total_samples) + ".npy", quad_intensity_nib)
                            np.save(sav_dir_imagesTs + str(total_samples) + ".npy", quad_intensity_nib)
                        else:
                            #np.save(sav_dir_imagesTr + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_' + str(total_samples) + ".npy", quad_intensity_nib)
                            np.save(sav_dir_imagesTr + str(total_samples) + ".npy", quad_intensity_nib)
                          
                        
                        """ When saving labels, must generate as labelled array along with json file describing the labels """
                        
                        if len(mask) == 0:  # already labelled if maskrcnn has done analysis so dont need to do this
                            labels = measure.label(quad_truth, connectivity=connectivity)    
                            
                            
                        else: 
                            labels = quad_truth  ### already labelled so dont need to do it again



                        labels = np.asarray(labels, dtype=int)
                                                
                        # if len(np.unique(quad_truth)) == 1:   ### if image is empty, do NOT run watershed, MATLAB binarization algorithm behaves weirdly
                        #                                       ### will set all of background to 1
                        #     labels = measure.label(quad_truth, connectivity=connectivity)                          
                        #     labels = np.asarray(labels, dtype=np.uint16)
                            
                        #     print('empty')
                            
                            
                        # elif water:
    
                        #     """ OPTIONAL: add watershed segmentation
                            
                        #             ***PROBLEM: will also delete small cells... so don't want this
                            
                        #     """
                        #     # Now we want to separate the two objects in image
                        #     # Generate the markers as local maxima of the distance to the background
                        #     # from scipy import ndimage as ndi
                        #     # from skimage.segmentation import watershed
                        #     # from skimage.feature import peak_local_max
                        #     # distance = ndi.distance_transform_edt(quad_truth)
                        #     # coords = peak_local_max(distance, footprint=np.ones((3, 10, 10)), labels=quad_truth,
                        #     #                         min_distance=1)  ### some other params here too, like threshold_abs
                        #     # mask = np.zeros(distance.shape, dtype=bool)
                        #     # mask[tuple(coords.T)] = True
                        #     # markers, _ = ndi.label(mask)
                        #     # labels = watershed(-distance, markers, mask=quad_truth)
                        #     # labels = np.asarray(labels, dtype=np.uint16)
                            
                            
                        #     # import napari
                        #     # viewer = napari.view_image(labels)
                        #     # #viewer.add_image(label)
                            
                            
                        #     # ### add cells back in that are missing
                        #     # missing = np.copy(quad_truth)
                        #     # missing[labels > 0] = 0
                        #     # missing = measure.label(missing)
                            
                        #     # missing = missing + np.max(labels)
                        #     # missing[missing == np.max(labels)] = 0
                            
                            
                        #     # labels = labels + missing
                        #     # labels = np.asarray(labels, dtype=np.uint16)s
                            
                            
                            
                        #     """ Do MATLAB watershed instead """
                        #     quad_truth[quad_truth > 0] = 255
                        #     quad_truth = np.asarray(quad_truth, dtype=np.uint8)
                        #     #quad_truth = np.moveaxis(quad_truth, 0, -1)
                        #     tiff.imwrite(sav_dir_WATER + str(total_samples) + "_watershed_seg.tif", quad_truth)
                            
                        #     # import matlab.engine
                        #     print('Running watershed in MATLAB')
                        #     # eng = matlab.engine.start_matlab()
                        #     # s = eng.genpath('./MATLAB_functions/')
                        #     # eng.addpath(s, nargout=0)
                        #     eng.main_Huganir_watershed_SEP_func(sav_dir_WATER, nargout=0)  ### EXPECTS IMAGE WITH binary value == 255
                        #     #eng.quit()
                        
                        #     ### read watershed image back in
                        #     images_w = glob.glob(os.path.join(sav_dir_WATER,'*.tif'))
                            
                        #     labels = tiff.imread(images_w[-1])
                        #     labels = np.asarray(labels, dtype=np.uint16)
                            
                            
                        #     ### then delete all the temporary files so can continue
                        #     for f in images_w:
                        #         os.remove(f)

                            
                            
                        # else:
                        #     labels = measure.label(quad_truth, connectivity=connectivity)                          
                        #     labels = np.asarray(labels, dtype=np.uint16)
                            
                            
                        # if total_samples == 36:
                        #     zzz
                        #     import napari
                        #     viewer = napari.view_image(labels)
                        #     #viewer.add_image(label)
                                                        



                        """ Clean up by removing edges, and then also small objects"""

                        def subtract_edges(labels, num_px, num_z=0):
                            labels[:, :num_px, :] = 0
                            labels[:, quad_size - num_px:, :] = 0
                            
    
                            labels[:, :, :num_px] = 0
                            labels[:, :, quad_size - num_px:] = 0                        
                            
                            # if also want to subtract z
                            if num_z > 0:
                                labels[:num_z, ...] = 0
                                labels[quad_depth - num_z:, ...] = 0       
                                
                                
                            return labels
                            
                        
                        
                        ### Re-number cells to start from 1
                        if len(mask) == 0:
                            labels = measure.label(labels, connectivity=connectivity)
                        else:
                            ### re-number every cell so it starts from 1
                            clean_labels = np.asarray(labels, dtype=int)
                            cc = measure.regionprops(labels)
                            
                            new_labels = np.zeros(np.shape(labels))
                            for cell_num, cell in enumerate(cc):
                                c = cell['coords']
                                new_labels[c[:, 0], c[:, 1], c[:, 2]] = cell_num + 1
                                
                        
                            labels = new_labels
            
                        # get labels PRIOR to edge subtraction so can add it back in after
                        labels = np.asarray(labels, dtype=int)
                        cc_save = measure.regionprops(labels)
                            
                        
                        
                            
                        labels_edge = subtract_edges(np.copy(labels), num_px=2, num_z=0)
                        
                        
  
                        
                        #labels = labels_edge

                            
                            
                        """ CLEAN UP LABELS BY REMOVING SMALL OBJECTS - dont do this for actual cleaned data """
                        
                        cc = measure.regionprops(labels_edge)
                        radii = []
                        too_small = 0
                        num_objs = 0
                        clean_labels = np.zeros(np.shape(labels_edge))
                        for cell in cc:
                            coords = cell['coords']
                            if cell['area'] >= min_size:
                                clean_labels[coords[:, 0], coords[:, 1], coords[:, 2]] = labels_edge[coords[:, 0], coords[:, 1], coords[:, 2]] 
                                
                                num_objs += 1
                            else:
                                too_small += 1
                        
                                
                        print('Num objects too small: ' + str(too_small))
                        print('Num objects: ' + str(num_objs))
                        
                        
                        
                        
                        ### ADD BACK-IN all objects that were NOT subtracted out
                        saved = np.unique(clean_labels)
                        saved = saved[saved != 0]
                        saved = saved - 1 ### get index to start from 0
                        
                        #cc_addback = cc_save[saved]
                        cleaned = np.zeros(np.shape(labels))
                        for cell_id, obj_id in enumerate(saved):
                            obj_id = int(obj_id)
                            coords = cc_save[obj_id]['coords']
                            cleaned[coords[:, 0], coords[:, 1], coords[:, 2]] = cell_id + 1
                        
                        labels = cleaned
                        
                        
                        
                                
 
                        
                        
                        

                        """ dilates image by a spherical ball of size radius """
                        import skimage

                        def dilate_by_ball_to_grayscale(input_im, radius):
                              ball_obj = skimage.morphology.ball(radius=radius)
                              input_im = skimage.morphology.dilation(input_im, footprint=ball_obj)  
                              #input_im[input_im > 0] = 1
                              return input_im

                        def erode_by_ball_to_grayscale(input_im, radius):
                              ball_obj = skimage.morphology.ball(radius=radius)
                              input_im = skimage.morphology.erosion(input_im, footprint=ball_obj)  
                              #input_im[input_im > 0] = 1
                              return input_im

                        
                        """ dilates image by a spherical ball of size radius """
                        def dilate_by_disk_to_binary(input_im, radius):
                              ball_obj = skimage.morphology.disk(radius=radius)
                              for s_id in range(len(input_im)):
                                  input_im[s_id] = skimage.morphology.dilation(input_im[s_id], footprint=ball_obj)  
                              return input_im

                        def erode_by_disk_to_binary(input_im, radius):
                              ball_obj = skimage.morphology.disk(radius=radius)
                              for s_id in range(len(input_im)):
                                  input_im[s_id] = skimage.morphology.erosion(input_im[s_id], footprint=ball_obj)  
                              return input_im



                        # if len(mask) == 0:
                        #     labels = measure.label(clean_labels, connectivity=connectivity)
                        # else:
                        #     ### re-number every cell so it starts from 1
                        #     clean_labels = np.asarray(clean_labels, dtype=int)
                        #     cc = measure.regionprops(clean_labels)
                            
                        #     new_labels = np.zeros(np.shape(clean_labels))
                        #     for cell_num, cell in enumerate(cc):
                        #         c = cell['coords']
                        #         new_labels[c[:, 0], c[:, 1], c[:, 2]] = cell_num + 1
                                
                        
                        #     labels = new_labels
            
            
            
            
            
    
                        labels = np.asarray(labels, dtype=np.uint16)

                        ### Only do this for original data, not for corrected data
                        if len(mask) == 0 and len(np.unique(labels)) > 1:
                            labels = dilate_by_disk_to_binary(np.copy(labels), radius=1)
                            labels = erode_by_disk_to_binary(labels, radius=1)
                        
                        
                        

            
                                            
                
                        cc = measure.regionprops(labels)
                        radii = []
                        for cell in cc:
                            radii.append(np.asarray(float(cell['equivalent_diameter'])))
                            
                            
                        

                        quad_truth_nib = np.moveaxis(labels, 0, -1)
                        #quad_truth_nib = nib.Nifti1Image(quad_truth_nib, affine=np.eye(4))
                        
                        
                
                        #zzz
                        if rand_int == 10:
                            #np.save(sav_dir_imagesTs + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) + '_' + str(total_samples) + "_seg.npy", quad_truth_nib)
                            np.save(sav_dir_imagesTs + str(total_samples) + "_seg.npy", quad_truth_nib)
                        
                        else:
                            #np.save(sav_dir_imagesTr + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) + '_' + str(total_samples) + "_seg.npy", quad_truth_nib)
                            np.save(sav_dir_imagesTr + str(total_samples) + "_seg.npy", quad_truth_nib)
                            


                        ### For debugging:
                        # tiff.imwrite(sav_dir_imagesTs + str(total_samples) + "_input_im.tif", np.moveaxis(quad_intensity_nib, -1, 0))    
                        # tiff.imwrite(sav_dir_imagesTs + str(total_samples) + "_seg.tif", np.moveaxis(quad_truth_nib, -1, 0))    

                        

                        ### FOR SAVING info_df.pickle
                        ### there is only 1 class (Oligos) so just set length of unique - 1 (for background)
                        num_objects = len(np.unique(quad_truth_nib)) - 1
                        
                        all_num_objs.append(num_objects)
                        class_ids = np.ones(num_objects).astype(np.int64)
                        fg_slices = np.where(np.sum(np.sum(quad_truth_nib, axis=0), axis=0))[0].astype(np.int64)
                        pid = str(total_samples)
                        
                        if rand_int == 10:
                            out_dir = sav_dir_imagesTs
                            
                            extra_dirs = '/OL_data/Tiger/'
                            #path = sav_dir_imagesTs + extra_dirs + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_' + str(total_samples) + ".npy"
                            path = sav_dir_imagesTs + extra_dirs + str(total_samples) + ".npy"
                        
                            info = {'out_dir':out_dir, 'path':path, 'class_ids':class_ids,
                                    'fg_slices':fg_slices, 'pid':pid, 
                                    'regression_vectors': radii, 'undistorted_rg_vectors': radii}
                            info_dict_test.append(info)
                        
                            
                        else:
                            out_dir = sav_dir_imagesTr
                            #path = sav_dir_imagesTr + extra_dirs + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_' + str(total_samples) + ".npy"
                            path = sav_dir_imagesTr + extra_dirs + str(total_samples) + ".npy"
                                 
                            info = {'out_dir':out_dir, 'path':path, 'class_ids':class_ids,
                                    'fg_slices':fg_slices, 'pid':pid,
                                    'regression_vectors': radii, 'undistorted_rg_vectors': radii}
                            info_dict_train.append(info)
                        

                        
                        # all_instances = np.unique(labels)
                        # dict_instances = {}
                        # for i in all_instances: 
                        #     if i == 0:   ### SKIP BACKGROUND
                        #         continue
                        #     dict_instances[str(i)] = 0   ### class is 0, which is oligos
                            
                          
                        # json_dict = {
                        #         "instances": dict_instances
                        #     }
                            
                                
                        
                        # # Serializing json  
                        # json_object = json.dumps(json_dict, indent = 4) 
                        # #print(json_object)
                        
                        # # Writing to sample.json
                        # if rand_int == 10:
                        #     with open(sav_dir_labelsTs + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_' + str(total_samples) + ".json", "w") as outfile:
                        #         outfile.write(json_object)
                        # else:
                        #     with open(sav_dir_labelsTr + filename + str(int(x)) + '_' + str(int(y)) + '_' + str(int(z)) +'_' + str(total_samples) + ".json", "w") as outfile:
                        #         outfile.write(json_object)
                        
                        
                        # if total_samples == 175:
                        #     zzz
                            
                        total_samples += 1
                        print(total_samples)
                        
                        
                        # if x > 200 and y > 200 and z > 20:
                        #     zzz
                        
                        # if total_samples > 200:
                        #     zzz

#eng.quit()

### PICKLE PROTOCOL doesn't matter, it's the pandas version that is mismatched that creates weird pickles



""" How to do this:
    
        use Python 3.8 normal Spyder to run all the code above (so skimage can load morphology module ect...)
        
        Then, run the command below to save.
        
        Then use other console to open up Spyder in Python 3.7 with RegRCNN2 virtualenv
        Then run the final commands to open up the pickle and then save it again
    
    """


plt.figure(); plt.hist(all_num_objs)
max_instances = np.max(all_num_objs)
print(max_instances)

########### if need to save and then reopen below

with open(sav_dir_imagesTr + 'info_df.pickle', 'wb') as handle:
    pickle.dump(info_dict_train, handle, protocol=4)
    
with open(sav_dir_imagesTs + 'info_df.pickle', 'wb') as handle:
    pickle.dump(info_dict_test, handle, protocol=4)
    
    
#zzz


########### ### open backup and convert to dataframe

with open(sav_dir_imagesTr + 'info_df.pickle', 'rb') as handle:
    info_dict_tr = pickle.load(handle)

with open(sav_dir_imagesTs + 'info_df.pickle', 'rb') as handle:
    info_dict_ts = pickle.load(handle)




info_df_tr = pd.DataFrame(info_dict_tr)
info_df_tr.to_pickle(sav_dir_imagesTr + 'info_df.pickle')

info_df_ts = pd.DataFrame(info_dict_ts)
info_df_ts.to_pickle(sav_dir_imagesTs + 'info_df.pickle')




