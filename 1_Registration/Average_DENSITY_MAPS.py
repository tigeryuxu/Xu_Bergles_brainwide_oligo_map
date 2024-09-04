#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:03:32 2023

@author: user
"""
import z5py

import glob, os
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
import pandas as pd
# pd.options.mode.chained_assignment = None  ### disable warning

import tifffile as tiff

import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean

from skimage.measure import label, regionprops, regionprops_table

import json
import matplotlib.pyplot as plt
    
from postprocess_func import *
import seaborn as sns
from scipy import stats   


import sys
sys.path.append("..")

from get_brain_metadata import *



# # All P60 brains
# exp_str = 'P60_rad5'
# list_brains = get_metadata(mouse_num = ['M127', 'M229', 'M126',  'M254', 'M256', 'M260']) #'M223', 'M299'])


# P240
# exp_str = 'P240_rad5'
# list_brains = get_metadata(mouse_num = ['M279', 'M286', 'M281', 'M285'])

# P620
# exp_str = 'P620_rad5'
# list_brains = get_metadata(mouse_num = ['M334', 'M97', 'M91', 'M271'])


# P800
# exp_str = 'P800_rad5'
# list_brains = get_metadata(mouse_num = ['Otx6', '5Otx5']) #'Otx18',]) #'1Otx7'])


# # Cup
# exp_str = 'CUPRIZONE_rad5'
# list_brains = get_metadata(mouse_num = ['M265', 'M266', 'M267']) 

# Cup RECOVERY
# exp_str = 'RECOVERY_rad5'
# list_brains = get_metadata(mouse_num = ['M310', 'M312', 'M313'])#, 'M311']) 



# # FVB/CD1
# exp_str = 'FVB_rad5'
# list_brains = get_metadata(mouse_num = ['M147', 'M155', 'M152']) 

# exp_str = 'CD1_rad5'
# list_brains = get_metadata(mouse_num = ['M170', 'M172']) 

# large = False

##################################################################################### For Large OL density maps

large = True

# All P60 brains
exp_str = 'P60_LARGE'
list_brains = get_metadata(mouse_num = ['M127',  'M126',  'M254', 'M299']) #'M223',, 'M229', M260])
## missing M256? 'M260' - currently re-running


# # 8 mos brains
# exp_str = 'P240_LARGE'
# list_brains = get_metadata(mouse_num = ['M279', 'M286', 'M285'])  ### Exclude because delip, not CUBIC 'M281', 

# # 22 mos brains
# exp_str = 'P620_LARGE'
# list_brains = get_metadata(mouse_num = [ 'M91', 'M334']) #(current)'M271',   ### EXCLUDE M271 for now???  'M97', 


# 30 mos brains
### EXCLUDE Otx6 for now???
# exp_str = 'P850_LARGE'
# list_brains = get_metadata(mouse_num = ['1Otx7', 'Otx18', '5Otx5']) #   'Otx6'   (current)


# # Cup
# exp_str = 'CUPRIZONE_LARGE'
# list_brains = get_metadata(mouse_num = ['M265', 'M266', 'M267']) 

# # Cup RECOVERY
# exp_str = 'RECOVERY_LARGE'
# list_brains = get_metadata(mouse_num = ['M310', 'M312', 'M313'])#, 'M311']) 





ANTS = 1


#%% Parse the json file so we can choose what we want to extract or mask out

#reference_atlas = '/home/user/.brainglobe/633_princeton_mouse_20um_v1.0/annotation.tiff'

# reference_atlas = '/home/user/.brainglobe/633_perens_lsfm_mouse_20um_v1.0/annotation.tiff'

reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation.tiff'

sav_fold = '/media/user/8TB_HDD/Mean_autofluor/'


ref_atlas = tiff.imread(reference_atlas)
ref_atlas = np.asarray(ref_atlas, dtype=np.uint32)

with open('../atlas_ids/atlas_ids.json') as json_file:
    data = json.load(json_file)
 
     
data = data['msg'][0]

#%% Also extract isotropic volumes from reference atlas DIRECTLY - to avoid weird expansion factors
print('Extracting main key volumes')
keys_dict = get_ids_all(data, all_keys=[], keywords=[''])  
main_keys = pd.DataFrame.from_dict(keys_dict)
main_keys = get_atlas_isotropic_vols(main_keys, ref_atlas, atlas_side='_W', XY_res=20, Z_res=20)
main_keys['atlas_vol_R'] = main_keys['atlas_vol_W']/2
main_keys['atlas_vol_L'] = main_keys['atlas_vol_W']/2

keys_df = main_keys.copy(deep=True)

#%% Loop through each experiment
all_autofluor = []
for exp in list_brains:
   

    input_path = exp['path']
    name_str = exp['name']
    
    n5_file = input_path + name_str + '.n5'
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'

    
    """ Loop through all the folders and do the analysis!!!"""
    #filename = n5_file.split('/')[-2]
    filename = n5_file.split('/')[-1].split('.')[0:-1]
    filename = '.'.join(filename) 
    sav_dir = input_path + '/' + filename + '_postprocess'
    
    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("\nSave directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("\nSave directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    

    #%% Load registered atlas
    print('Loading density map')

    if not large:
        reg_autofluor = tiff.imread(sav_dir + name_str + '_DENSITY_MAP_CERE_rad5_minsize30.tif')

    else:
        reg_autofluor = tiff.imread(sav_dir + name_str + '_LARGE_DENSITY_MAP.tif')   
    
    all_autofluor.append(reg_autofluor)
    
all_autofluor = np.asarray(all_autofluor)
mean_autofluor = np.mean(all_autofluor, axis=0)
mean_autofluor = np.asarray(mean_autofluor, dtype=np.uint32)


      


#%% Set all zero cell areas to value 1 so no holes in density map
### First start by defining everything we do NOT want

cc_allen = regionprops(ref_atlas, cache=False)
cc_labs_allen = [region['label'] for region in cc_allen]
cc_labs_allen = np.asarray(cc_labs_allen)

    
    
all_sub_id = []

### just do striatum
sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Striatum')
all_sub_id.append(sub_idx)

sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Pallidum')
all_sub_id.append(sub_idx)

sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='fiber tracts')
all_sub_id.append(sub_idx)
sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Cerebellum')
all_sub_id.append(sub_idx)
sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Hindbrain')
all_sub_id.append(sub_idx)
sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='ventricular systems')
all_sub_id.append(sub_idx)
sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Main olfactory bulb')
all_sub_id.append(sub_idx)
sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Anterior olfactory nucleus')
all_sub_id.append(sub_idx)


# sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Piriform area')
# all_sub_id.append(sub_idx)
# sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Nucleus of the lateral olfactory tract')
# all_sub_id.append(sub_idx)


### optional midbrain
sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Midbrain')
all_sub_id.append(sub_idx)


sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Interbrain')
all_sub_id.append(sub_idx)

all_sub_id = [x for xs in all_sub_id for x in xs]

all_sub_id.append(0)        ### for "root" 
all_sub_id.append(np.where(keys_df['names'] == 'Olfactory areas')[0][0]) ### add olfactory areas
all_sub_id.append(np.where(keys_df['names'] == 'Accessory olfactory bulb, mitral layer')[0][0]) ### add olfactory areas
all_sub_id.append(np.where(keys_df['names'] == 'Accessory olfactory bulb, granular layer')[0][0]) ### add olfactory areas
all_sub_id.append(np.where(keys_df['names'] == 'Accessory olfactory bulb, glomerular layer')[0][0]) ### add olfactory areas

sub_keys = keys_df.iloc[all_sub_id]
sub_keys.reset_index(inplace=True, drop=True)


sub_ids = np.asarray(sub_keys['ids'])

""" Mask out just the layer using Allen atlas """
### do for Allen atlas
remove_regions = np.zeros(np.shape(ref_atlas))
for idx in sub_ids:
    cur_id = np.where(cc_labs_allen == idx)[0]
    
    #print(cur_id)
    if len(cur_id) == 0:  ### if it does not exists in atlas
        continue
    cur_coords = cc_allen[cur_id[0]]['coords']
    remove_regions[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = idx
   

iso_layer = np.copy(ref_atlas)
iso_layer[remove_regions > 0] = 0   ### delete all other regions
# iso_layer[remove_regions == 0] = 0  ### keep current region

iso_layer[iso_layer > 0] = 1


bw_mean = np.copy(mean_autofluor)
bw_mean[bw_mean > 0] = 1

leftover = iso_layer - bw_mean

mean_autofluor[leftover > 0] = 1



#%% SAVE


tiff.imwrite(sav_fold + exp_str + '_mean_density_MAP.tif', mean_autofluor)



if large:    

    #   %% Mask out fiber tracks
    ### First start by defining everything we do NOT want
    all_sub_id = []
    
    ### just do striatum
    sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Isocortex')
    # all_sub_id.append(sub_idx)
    
    
    
    layer6b = keys_df.iloc[sub_idx][keys_df.iloc[sub_idx]['names'].str.contains('6b')]

    # all_sub_id = [x for xs in all_sub_id for x in xs]

    
    # sub_keys = keys_df.iloc[all_sub_id]
    # sub_keys.reset_index(inplace=True, drop=True)
    
    
    sub_ids = np.asarray(layer6b['ids'])
    
    """ Mask out just the layer using Allen atlas """
    ### do for Allen atlas
    remove_regions = np.zeros(np.shape(ref_atlas))
    for idx in sub_ids:
        cur_id = np.where(cc_labs_allen == idx)[0]
        
        #print(cur_id)
        if len(cur_id) == 0:  ### if it does not exists in atlas
            continue
        cur_coords = cc_allen[cur_id[0]]['coords']
        remove_regions[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = idx
       
    
    # cortex_only = np.copy(ref_atlas)
    # cortex_only[remove_regions > 0] = 0   ### delete all other regions
    
    
    cortex_mean = np.copy(mean_autofluor)
    cortex_mean[remove_regions > 0] = 0
    
    
    
    
    
    
    tiff.imwrite(sav_fold + exp_str + '_mean_density_MAP_CORTEX_no_6b.tif', cortex_mean)
    
    


if not large:
    #   %% Mask out fiber tracks
    ### First start by defining everything we do NOT want
    
    cc_allen = regionprops(ref_atlas, cache=False)
    cc_labs_allen = [region['label'] for region in cc_allen]
    cc_labs_allen = np.asarray(cc_labs_allen)
    
        
    all_sub_id = []
    
    ### just do striatum
    sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Isocortex')
    all_sub_id.append(sub_idx)
    
    
    
    
    
    # sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='fiber tracts')
    # all_sub_id.append(sub_idx)
    # sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Cerebellum')
    # all_sub_id.append(sub_idx)
    # sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Hindbrain')
    # all_sub_id.append(sub_idx)
    # sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='ventricular systems')
    # all_sub_id.append(sub_idx)
    # sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Main olfactory bulb')
    # all_sub_id.append(sub_idx)
    # sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Anterior olfactory nucleus')
    # all_sub_id.append(sub_idx)
    
    # ### optional midbrain
    # sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Midbrain')
    # all_sub_id.append(sub_idx)
    
    all_sub_id = [x for xs in all_sub_id for x in xs]
    
    # all_sub_id.append(0)        ### for "root" 
    # all_sub_id.append(np.where(keys_df['names'] == 'Olfactory areas')[0][0]) ### add olfactory areas
    # all_sub_id.append(np.where(keys_df['names'] == 'Accessory olfactory bulb, mitral layer')[0][0]) ### add olfactory areas
    # all_sub_id.append(np.where(keys_df['names'] == 'Accessory olfactory bulb, granular layer')[0][0]) ### add olfactory areas
    # all_sub_id.append(np.where(keys_df['names'] == 'Accessory olfactory bulb, glomerular layer')[0][0]) ### add olfactory areas
    
    
    sub_keys = keys_df.iloc[all_sub_id]
    sub_keys.reset_index(inplace=True, drop=True)
    
    
    sub_ids = np.asarray(sub_keys['ids'])
    
    """ Mask out just the layer using Allen atlas """
    ### do for Allen atlas
    remove_regions = np.zeros(np.shape(ref_atlas))
    for idx in sub_ids:
        cur_id = np.where(cc_labs_allen == idx)[0]
        
        #print(cur_id)
        if len(cur_id) == 0:  ### if it does not exists in atlas
            continue
        cur_coords = cc_allen[cur_id[0]]['coords']
        remove_regions[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = idx
       
    
    cortex_only = np.copy(ref_atlas)
    cortex_only[remove_regions == 0] = 0   ### delete all other regions
    
    
    cortex_mean = np.copy(mean_autofluor)
    cortex_mean[remove_regions == 0] = 0
    tiff.imwrite(sav_fold + exp_str + '_mean_density_MAP_CORTEX_ONLY.tif', cortex_mean)
    
    
    
    
    #%% Get just Hippocampus
    
    all_sub_id = []
    ### just do striatum
    sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Hippocampal formation')
    all_sub_id.append(sub_idx)
    
    all_sub_id = [x for xs in all_sub_id for x in xs]
    
    
    sub_keys = keys_df.iloc[all_sub_id]
    sub_keys.reset_index(inplace=True, drop=True)
    
    
    sub_ids = np.asarray(sub_keys['ids'])
    
    """ Mask out just the layer using Allen atlas """
    ### do for Allen atlas
    remove_regions = np.zeros(np.shape(ref_atlas))
    for idx in sub_ids:
        cur_id = np.where(cc_labs_allen == idx)[0]
        
        #print(cur_id)
        if len(cur_id) == 0:  ### if it does not exists in atlas
            continue
        cur_coords = cc_allen[cur_id[0]]['coords']
        remove_regions[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = idx
       
    
    cortex_only = np.copy(ref_atlas)
    cortex_only[remove_regions == 0] = 0   ### delete all other regions
    
    
    cortex_mean = np.copy(mean_autofluor)
    cortex_mean[remove_regions == 0] = 0
    tiff.imwrite(sav_fold + exp_str + '_mean_density_MAP_HIPPOCAMPAL_ONLY.tif', cortex_mean)
    
    
    
    
    #%% Get just Cerebellum
    
    all_sub_id = []
    ### just do striatum
    sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Cerebellum')
    all_sub_id.append(sub_idx)
    
    sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='arbor vitae')
    all_sub_id.append(sub_idx)
    
    all_sub_id = [x for xs in all_sub_id for x in xs]
    
    
    sub_keys = keys_df.iloc[all_sub_id]
    sub_keys.reset_index(inplace=True, drop=True)
    
    
    sub_ids = np.asarray(sub_keys['ids'])
    
    """ Mask out just the layer using Allen atlas """
    ### do for Allen atlas
    remove_regions = np.zeros(np.shape(ref_atlas))
    for idx in sub_ids:
        cur_id = np.where(cc_labs_allen == idx)[0]
        
        #print(cur_id)
        if len(cur_id) == 0:  ### if it does not exists in atlas
            continue
        cur_coords = cc_allen[cur_id[0]]['coords']
        remove_regions[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = idx
       
    
    cortex_only = np.copy(ref_atlas)
    cortex_only[remove_regions == 0] = 0   ### delete all other regions
    
    
    cortex_mean = np.copy(mean_autofluor)
    cortex_mean[remove_regions == 0] = 0
    tiff.imwrite(sav_fold + exp_str + '_mean_density_MAP_Cerebellum_ONLY.tif', cortex_mean)
    
    
    



