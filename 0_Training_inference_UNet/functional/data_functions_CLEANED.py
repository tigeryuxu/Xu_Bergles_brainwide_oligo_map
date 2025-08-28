# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:25:15 2017

@author: Tiger
"""

""" Retrieves validation images
"""

import numpy as np
#from PIL import Image
from os import listdir
from os.path import isfile, join
from skimage import measure
from natsort import natsort_keygen, ns
import os
import pickle
import scipy.io as sio
from tifffile import imsave

#import zipfile
#import bz2

#from plot_functions_CLEANED import *
#from data_functions import *
#from post_process_functions import *
#from UNet import *

#from skan import skeleton_to_csgraph
from skimage.morphology import skeletonize_3d, skeletonize
import skimage




""" Check size limits """
def check_limits(all_neighborhoods, width_tmp, height_tmp, depth_tmp):
    
        """ Make sure nothing exceeds size limits """
        idx = 0; 
        for neighbor_be in all_neighborhoods:
            
            if len(neighbor_be) > 0:
                if np.any(neighbor_be[:, 0] >= width_tmp):
                    all_neighborhoods[idx][np.where(neighbor_be[:, 0] >= width_tmp), 0] = width_tmp - 1
                    
                if np.any(neighbor_be[:, 1] >= height_tmp):
                    all_neighborhoods[idx][np.where(neighbor_be[:, 1] >= height_tmp), 1] = height_tmp - 1
    
                if np.any(neighbor_be[:, 2] >= depth_tmp):
                    all_neighborhoods[idx][np.where(neighbor_be[:, 2] >= depth_tmp), 2] = depth_tmp - 1
            idx += 1
            
        return all_neighborhoods   

def check_limits_single(all_neighborhoods, width_tmp, height_tmp, depth_tmp):
    
        """ Make sure nothing exceeds size limits """
        idx = 0; 
        for neighbor_be in all_neighborhoods:
            
            if len(neighbor_be) > 0:
                if np.any(neighbor_be[0] >= width_tmp):
                    all_neighborhoods[idx][0] = width_tmp - 1
                    
                if np.any(neighbor_be[1] >= height_tmp):
                    all_neighborhoods[idx][1] = height_tmp - 1
    
                if np.any(neighbor_be[2] >= depth_tmp):
                    all_neighborhoods[idx][2] = depth_tmp - 1
            idx += 1
            
        return all_neighborhoods  
    
    

""" dilates image by a spherical ball of size radius """
def erode_by_ball_to_binary(input_im, radius):
     ball_obj = skimage.morphology.ball(radius=radius)
     input_im = skimage.morphology.erosion(input_im, selem=ball_obj)  
     input_im[input_im > 0] = 1
     return input_im

""" dilates image by a spherical ball of size radius """
def dilate_by_ball_to_binary(input_im, radius):
     ball_obj = skimage.morphology.ball(radius=radius)
     input_im = skimage.morphology.dilation(input_im, selem=ball_obj)  
     input_im[input_im > 0] = 1
     return input_im

""" dilates image by a spherical ball of size radius """
def dilate_by_disk_to_binary(input_im, radius):
     ball_obj = skimage.morphology.disk(radius=radius)
     input_im = skimage.morphology.dilation(input_im, selem=ball_obj)  
     input_im[input_im > 0] = 1
     return input_im

""" dilates image by a cube of size width """
def dilate_by_cube_to_binary(input_im, width):
     cube_obj = skimage.morphology.cube(width=width)
     input_im = skimage.morphology.dilation(input_im, selem=cube_obj)  
     input_im[input_im > 0] = 1
     return input_im

""" erodes image by a cube of size width """
def erode_by_cube_to_binary(input_im, width):
     cube_obj = skimage.morphology.cube(width=width)
     input_im = skimage.morphology.erosion(input_im, selem=cube_obj)  
     input_im[input_im > 0] = 1
     return input_im


""" Applies CLAHE to a 2D image """           
# def apply_clahe_by_slice(crop, depth):
#      clahe_adjusted_crop = np.zeros(np.shape(crop))
#      for slice_idx in range(depth):
#           slice_crop = np.asarray(crop[:, :, slice_idx], dtype=np.uint8)
#           adjusted = equalize_adapthist(slice_crop, kernel_size=None, clip_limit=0.01, nbins=256)
#           clahe_adjusted_crop[:, :, slice_idx] = adjusted
                 
#      crop = clahe_adjusted_crop * 255
#      return crop

""" Take input bw image and returns coordinates and degrees pixel map, where
         degree == # of pixels in nearby CC space
                 more than 3 means branchpoint
                 == 2 means skeleton normal point
                 == 1 means endpoint
         coordinates == z,x,y coords of the full skeleton object
         
    *** works for 2D and 3D inputs ***
"""

def bw_skel_and_analyze(bw):
     if bw.ndim == 3:
          skeleton = skeletonize_3d(bw)
     elif bw.ndim == 2:
          skeleton = skeletonize(bw)
     skeleton[skeleton > 0] = 1
    
     
     if skeleton.any() and np.count_nonzero(skeleton) > 1:
          try:
               pixel_graph, coordinates, degrees = skeleton_to_csgraph(skeleton)
          except:
               pixel_graph = np.zeros(np.shape(skeleton))
               coordinates = []
               degrees = np.zeros(np.shape(skeleton))               
     else:
          pixel_graph = np.zeros(np.shape(skeleton))
          coordinates = []
          degrees = np.zeros(np.shape(skeleton))
          
     return degrees, coordinates


""" removes detections on the very edges of the image """
def clean_edges(im, extra_z=1, extra_xy=3, skip_top=0):
     im_size = np.shape(im);
     w = im_size[1];  h = im_size[2]; depth = im_size[0];
     labelled = measure.label(im)
     cc_coloc = measure.regionprops(labelled)
    
     cleaned_im = np.zeros(np.shape(im))
     for obj in cc_coloc:
         coords = obj['coords']
         
         bool_edge = 0
         for c in coords:
              if ((c[0] <= 0 + extra_z and not skip_top) or c[0] >= depth - extra_z):
                   bool_edge = 1
                   break;
              if (c[1] <= 0 + extra_xy or c[1] >= w - extra_xy):
                   bool_edge = 1
                   break;                                       
              if (c[2] <= 0 + extra_xy or c[2] >= h - extra_xy):
                   bool_edge = 1
                   break;                                        

         if not bool_edge:
              for obj_idx in range(len(coords)):
                   cleaned_im[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1

     return cleaned_im                           
 

def find_TP_FP_FN_from_im(seg_train, truth_im):

     coloc = seg_train + truth_im
     bw_coloc = coloc > 0
     labelled = measure.label(truth_im)
     cc_coloc = measure.regionprops(labelled, intensity_image=coloc)
     
     true_positive = np.zeros(np.shape(coloc))
     TP_count = 0;
     FN_count = 0;
     for obj in cc_coloc:
          max_val = obj['max_intensity']
          #coords = obj['coords']
          if max_val > 1:
               TP_count += 1
               #for obj_idx in range(len(coords)):
               #     true_positive[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1
          else:
               FN_count += 1
 
     
     FP_count = 0;
     labelled = measure.label(bw_coloc)
     cc_coloc = measure.regionprops(labelled, intensity_image=coloc)
     for obj in cc_coloc:
          max_val = obj['max_intensity']
          #coords = obj['coords']
          if max_val == 1:
               FP_count += 1 
              
     return TP_count, FN_count, FP_count

           

def find_TP_FP_FN_from_seg(segmentation, truth_im, size_limit=0):
     seg = segmentation      
     true = truth_im  
     
     """ Also remove tiny objects from Truth due to error in cropping """
     labelled = measure.label(true)
     cc_coloc = measure.regionprops(labelled)
     
     cleaned_truth = np.zeros(np.shape(true))
     for obj in cc_coloc:
          coords = obj['coords']
          
          # can also skip by size limit          
          if len(coords) > 10:
               for obj_idx in range(len(coords)):
                    cleaned_truth[coords[obj_idx,0], coords[obj_idx,1], coords[obj_idx,2]] = 1
    
     
     """ Find matched """
     coloc = seg + true
     bw_coloc = coloc > 0
     labelled = measure.label(true)
     cc_coloc = measure.regionprops(labelled, intensity_image=coloc)
     
     true_positive = np.zeros(np.shape(coloc))
     TP_count = 0;
     FN_count = 0;
     for obj in cc_coloc:
          max_val = obj['max_intensity']
          coords = obj['coords']
          
          # can also skip by size limit          
          if max_val > 1 and len(coords) > size_limit:
               TP_count += 1
          else:
               FN_count += 1
 
     
     FP_count = 0;
     labelled = measure.label(seg)
     cc_coloc = measure.regionprops(labelled, intensity_image=coloc)
     cleaned_seg = np.zeros(np.shape(seg))
     for obj in cc_coloc:
          max_val = obj['max_intensity']
          coords = obj['coords']
     
          # can also skip by size limit
          if  len(coords) < size_limit:
               continue;
          else:
               for obj_idx in range(len(coords)):
                    cleaned_seg[coords[obj_idx, 0], coords[obj_idx, 1], coords[obj_idx, 2]] = 1
          
          if max_val == 1:
               FP_count += 1 
              
     return TP_count, FN_count, FP_count, cleaned_truth, cleaned_seg





""" Convert voxel list to array """
def convert_vox_to_matrix(voxel_idx, zero_matrix):
    for row in voxel_idx:
        #print(row)
        zero_matrix[(row[0], row[1], row[2])] = 1
    return zero_matrix



""" converts a matrix into a multipage tiff to save!!! """
def convert_matrix_to_multipage_tiff(matrix):
    rolled = np.rollaxis(matrix, 2, 0).shape  # changes axis to be correct sizes
    tiff_image = np.zeros((rolled), 'uint8')
    for i in range(len(tiff_image)):
        tiff_image[i, :, :] = matrix[:, :, i]
        
    return tiff_image


""" Saving the objects """
def save_pkl(obj_save, s_path, name):
    with open(s_path + name, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([obj_save], f)

"""Getting back the objects"""
def load_pkl(s_path, name):
    with open(s_path + name, 'rb') as f:  # Python 3: open(..., 'rb')
      loaded = pickle.load(f)
      obj_loaded = loaded[0]
      return obj_loaded



"""
    To normalize by the mean and std
"""
def normalize_im(im, mean_arr, std_arr):
    normalized = (im - mean_arr)/std_arr 
    return normalized        
    




