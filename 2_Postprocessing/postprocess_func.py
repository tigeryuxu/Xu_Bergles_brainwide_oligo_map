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

import seaborn as sns


""" Cortex_means is used to do any sort of sorting,

        df_concat is used to actually plot things according to the names and sorting style from cortex_means
 """

def get_subkeys_to_plot(df_means, df_concat, reg_name='Isocortex', dname='density_W', to_remove='MO|SS', to_remove_substring='', lvl_low=5, lvl_high=9):

    sub_idx = get_sub_regions_atlas(df_means, child_id=[], sub_keys=[], reg_name=reg_name)
    
    sub_keys = df_means.iloc[sub_idx]
    
    ### REMOVE OVERARCHING AREAS
    sub_keys = sub_keys[sub_keys['acronym'].str.fullmatch(to_remove) == False]
    
    ### REMOVE ADDITIONAL REGIONS VIA SUBSTRING --- doesnt do anything because it's not summed up here for overarching regions
    # if len(to_remove_substring) > 0:
    #     print(to_remove_substring)
    #     sub_keys = sub_keys[sub_keys['names'].str.contains(to_remove_substring) == False]
    
    ### use above sub names to index df_means
    means = df_means[df_means['acronym'].isin(sub_keys['acronym'])]
    # means = means.dropna()
    
    means = means[means['density_W'].notna()]

    
    regions = means.iloc[np.where((means['st_level'] < lvl_high) & (means['st_level'] > lvl_low))[0]]
    regions = regions.sort_values(by=[dname], ascending=False, ignore_index=True)
    names_to_plot = regions['acronym']
    plot_vals = df_concat[df_concat['acronym'].isin(names_to_plot)]
    
    return plot_vals, names_to_plot
    
def boxplot_by_subkey(df_means, df_concat, reg_name='Isocortex', dname='density_W', to_remove='MO|SS', to_remove_substring='', lvl_low=5, lvl_high=9):
 
    plot_vals, names_to_plot = get_subkeys_to_plot(df_means, df_concat, reg_name, dname, to_remove, to_remove_substring, lvl_low, lvl_high)
    
    # Make boxplots of cortex by REGION
    plt.figure()
    sns.boxplot(x=plot_vals[dname], y=plot_vals['acronym'], order=names_to_plot)
    sns.stripplot(x=plot_vals[dname], y=plot_vals['acronym'], order=names_to_plot)
    ax = plt.gca()
    plt.yticks(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    plt.tight_layout()
    
    plt.savefig('regions_' + dname + '_' + reg_name + '.png', dpi=300)

def get_ids_of_layer_from_keys(keys_df, layer):
    ids = [i for i, s in enumerate(keys_df['names']) if layer in s]
    return ids

def get_sub_regions_atlas(keys_df, child_id, sub_keys, reg_name='Isocortex'):
    
    if reg_name:
        cur_id = np.where(keys_df['names'] == reg_name)[0][0]
        sub_keys.append(cur_id)
        
    else:
        
        if len(np.where(keys_df['ids'] == child_id)[0]) == 0:  ### means layer 6 was already deleted
            print(child_id)
            print('layer_6_deleted')
            return sub_keys

    
        cur_id = np.where(keys_df['ids'] == child_id)[0][0]
        sub_keys.append(cur_id)
        
    # print(cur_id)
    # for child in keys_df['children'][cur_id]:
        
    for child in keys_df.iloc[cur_id]['children']:
        sub_keys = get_sub_regions_atlas(keys_df, child_id=child, sub_keys=sub_keys, reg_name=None)
    
    return sub_keys
    

### Get by acronym
def get_sub_regions_by_acronym(keys_df, child_id, sub_keys, reg_name='Isocortex'):
    
    if reg_name:
        cur_id = np.where(keys_df['acronym'] == reg_name)[0][0]
        sub_keys.append(cur_id)
        
    else:
        
        if len(np.where(keys_df['ids'] == child_id)[0]) == 0:  ### means layer 6 was already deleted
            print('children dropped previously')
            return sub_keys

    
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
    ### TIGER - changed this to just getting the main isotropic volume from the reference atlas instead (to ignore expansion factors)
    keys_df = get_atlas_isotropic_vols(keys_df, atlas, atlas_side + '_relative', XY_res, Z_res)
        
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
     keys_dict['acronym'] = data['acronym']  
 
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
 