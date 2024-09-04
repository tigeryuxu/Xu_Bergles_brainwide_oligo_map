#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:58:36 2023

@author: user
"""

import pandas as pd

import tifffile as tiff

import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean

from skimage.measure import label, regionprops, regionprops_table

import json

import matplotlib.pyplot as plt
import SimpleITK as sitk



def scale_to_16_bits(img):
    """
    Normalise the input image to the full 0-2^16 bit depth.

    :param np.array img: The input image
    :return: The normalised image
    :rtype: np.array
    """
    normalised = img / img.max()
    return normalised * (2**16 - 1)


def scale_and_convert_to_16_bits(img):
    """
    Normalise the input image to the full 0-2^16 bit depth, and return as
    type: "np.uint16".

    :param np.array img: The input image
    :return: The normalised, 16 bit image
    :rtype: np.array
    """
    img = scale_to_16_bits(img)
    return img.astype(np.uint16, copy=False)

""" Run N4 correction """

def N4_correction(input_im, mask_im, shrinkFactor=1, numberFittingLevels=4):
    input_im = np.asarray(input_im, dtype=np.float32)
    inputImage = sitk.GetImageFromArray(input_im)
    image = sitk.Shrink(inputImage, [shrinkFactor] * inputImage.GetDimension())
    
    if mask_im is not None:
        mask_im = np.asarray(mask_im, dtype=np.uint8)
        maskImage = sitk.GetImageFromArray(mask_im)
        mask = sitk.Shrink(maskImage, [shrinkFactor] * maskImage.GetDimension())
        

    
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    
    
    ### default is [50, 50, 50, 50]   ---> 4 levels with 50 iterations each    
    # num_iterations = 4
    # corrector.SetMaximumNumberOfIterations(
    #     [num_iterations] * numberFittingLevels
    # )
    
    
    
    if mask_im is not None:
        corrected_image = corrector.Execute(image, mask)
    else:
        corrected_image = corrector.Execute(image)
    
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    
    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)
    
    #sitk.WriteImage(corrected_image_full_resolution)
    
    im = sitk.GetArrayFromImage(corrected_image_full_resolution)
    
    return im, log_bias_field



def get_ids_of_layer_from_keys(keys_df, layer):
    ids = [i for i, s in enumerate(keys_df['names']) if layer in s]
    return ids

def get_sub_regions_atlas(keys_df, child_id, sub_keys, reg_name='Isocortex'):
    
    if reg_name:
        cur_id = np.where(keys_df['names'] == reg_name)[0][0]
        sub_keys.append(cur_id)
        
    else:
        cur_id = np.where(keys_df['ids'] == child_id)[0][0]
        sub_keys.append(cur_id)
        
    for child in keys_df['children'][cur_id]:
        sub_keys = get_sub_regions_atlas(keys_df, child_id=child, sub_keys=sub_keys, reg_name=None)
    
    return sub_keys
    




"""
    Takes as input keys_df and search_ids, which are the ids corresponding to regions of interest
    
    ***plots them with values from "list_plot_vals"
"""

def query_and_plot(keys_df, search_ids, atlas, cc, cc_labs, list_plot_vals):
    atlas_ids = np.asarray(keys_df['ids'][search_ids])
    #density = np.asarray(keys_df['num_OLs_scaleddiff'][search_ids])
    
    plot_im = np.zeros(np.shape(atlas))
    for ix, id_plot in enumerate(atlas_ids):
        cc_id = np.where(cc_labs == id_plot)[0][0]
        coords = cc[cc_id]['coords']
        
        plot_im[coords[:, 0], coords[:, 1], coords[:, 2]] = list_plot_vals[ix]

    return plot_im




def plot_max(im, ax=0, plot=1):
     max_im = np.amax(im, axis=ax)
     if plot:
         plt.figure(); plt.imshow(max_im)
     
     return max_im
 
      

""" Make atlas isotropic and then grab volumes and scale to cubic millimeter instead of microns """
def get_atlas_isotropic_vols(keys_df, atlas, atlas_side, XY_res, Z_res):
    
    res_diff = XY_res/Z_res
    
    #always want to downsample!!!
    if res_diff < 1:
        atlas_isotropic = rescale(atlas, [1, res_diff, res_diff], anti_aliasing=False, order=0, preserve_range=True)   ### rescale the images
        new_res = Z_res
    else:
        atlas_isotropic = rescale(atlas, [1/res_diff, 1, 1], anti_aliasing=False, order=0, preserve_range=True)   ### rescale the images
        new_res = XY_res

    # get relative volume per region (num voxels)
    print('getting absolute volumes per region')
    cc = regionprops(atlas_isotropic, cache=False)
    keys_df['atlas_vol'+atlas_side] = np.nan
    for region in cc:
        
        id_reg = region['label']
        idloc = np.where(keys_df['ids'] == id_reg)[0]
        
        
        
        vol = len(region['coords'])
                
        scaled_vol = vol * pow(new_res, 3)  ### 1.843 um/px * 16 downsampling fold to the power of 3 (isotropic to cubic volume)
        scaled_vol = scaled_vol * pow(10, -9)  ### then scale from cubic micron to cubic millimeter
        
        
        keys_df.loc[idloc, 'atlas_vol'+atlas_side] = scaled_vol

        
    return keys_df
        
        
        

    
def cells_to_atlas_df(keys_df, coords_df, cell_pos, atlas, atlas_side, XY_res, Z_res, size_thresh=500):
    ids = atlas[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]]
    
    # get num cells per region
    print('matching cells to atlas-type: ' + atlas_side)
    num_cells = []
    num_large_cells = []
    for idx in keys_df['ids']:
        num = len(np.where(ids == idx)[0])
        num_cells.append(num)
        
        # also get large cells
        num_large = len(np.where((ids==idx) & (coords_df['vols'] > size_thresh))[0])
        num_large_cells.append(num_large)
    
    ### Get isotropic volume of atlas
    keys_df = get_atlas_isotropic_vols(keys_df, atlas, atlas_side, XY_res, Z_res)
        
    # Add all to dataframe
    keys_df['num_OLs' + atlas_side] = num_cells
    keys_df['num_large' + atlas_side] = num_large_cells
    keys_df['density'+ atlas_side] = keys_df['num_OLs'+atlas_side]/keys_df['atlas_vol'+atlas_side]
    
    return keys_df

"""
    Parse json file:
        
        get hierachy of atlas_ids --> so know what level of complexity you want?
        
        
"""
def get_ids_all(data, all_keys, keywords):
     keys_dict = {'ids': None, 'names': None, 'parent':None, 'st_level':None, 'children':[]}
     
     #if any(word in data['name'].casefold() for word in keywords):   ### check if any of the keywords appear in the name
     keys_dict['ids']      = data['id']
     keys_dict['names']    = data['name']
     keys_dict['parent']   = data['parent_structure_id']
     keys_dict['st_level'] = data['st_level']    
 
     all_children = []
     for child in data['children']:
          all_children.append(child['id'])
    
     keys_dict['children'] = all_children
     
     all_keys.append(keys_dict)
     
     # start recursion
     for child in data['children']:
          all_keys = get_ids_all(child, all_keys, keywords)
          
          
          
     return all_keys
 
    
 
    
"""
    Query for specific IDs
"""



def get_ids(data, keys_dict, keywords):
     
     if any(word in data['name'].casefold() for word in keywords):   ### check if any of the keywords appear in the name
          keys_dict['ids'].append(data['id'])
          keys_dict['names'].append([data['name']])
          
          row = {}
          
          
     
     for child in data['children']:
          
          keys_dict = get_ids(child, keys_dict, keywords)
          
          
     return keys_dict
 