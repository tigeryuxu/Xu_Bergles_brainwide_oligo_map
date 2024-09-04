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




# list_brains = get_metadata(mouse_num = ['5Otx5'])     ### correcting with MaMut (Fully done)
# list_brains = get_metadata(mouse_num = ['M312'])     ### correcting with MaMut
# list_brains = get_metadata(mouse_num = ['M265'])     ### correcting with MaMut


# list_brains = get_metadata(mouse_num = ['M334'])     ### correcting with MaMut

# list_brains = get_metadata(mouse_num = ['M91'])     ### correcting with MaMut (Fully done)


# list_brains = get_metadata(mouse_num = ['Otx6'])     ### correcting with MaMut (current)
# list_brains = get_metadata(mouse_num = ['Otx18'])     ### correcting with MaMut
# list_brains = get_metadata(mouse_num = ['M281'])     ### correcting with MaMut --- skip (smaller)


# list_brains = get_metadata(mouse_num = ['M256'])     ### correcting with MaMut
# list_brains = get_metadata(mouse_num = ['M127'])     ### correcting with MaMut --- skip (smaller)

# fully_tracked = False
# list_brains = get_metadata(mouse_num = ['M312', 'M265', 'M334', 'Otx6', 'Otx18', 'M281', 'M256', 'M127'])

### FULLY corrected
# fully_tracked = True
# list_brains = get_metadata(mouse_num = ['5Otx5', 'M91'])




### Skip M127 and M281 for cleaned data
# fully_tracked = False
# list_brains = get_metadata(mouse_num = ['M312', 'M265', 'M334', 'Otx6', 'Otx18', 'M256']) #'M281', ' M127'])

### FULLY corrected
fully_tracked = True
list_brains = get_metadata(mouse_num = ['5Otx5', 'M91'])



#%% Loop through each experiment
for exp in list_brains:

    input_path = exp['path']
    name_str = exp['name']
    
    n5_file = input_path + name_str + '.n5'
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'


    """ Loop through all the folders and do the analysis!!!"""
    filename = n5_file.split('/')[-1].split('.')[0:-1]
    filename = '.'.join(filename) 
    sav_dir = input_path + '/' + filename + '_postprocess'
    
    
    """ For testing ILASTIK images """
    images = glob.glob(os.path.join(input_path,'*.csv'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i.replace('_df.pkl','_input_im.tif'),pkl=i, segmentation=i.replace('_df.pkl','_segmentation_overlap3.tif'),
                     shifted=i.replace('_df.pkl', '_shifted.tif')) for i in images]
     
    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("\nSave directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("\nSave directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    


    
    #%% Only do gray matter comparisons to start
    print('Converting coordinates to correct scale')
    ### supposedly atlass is at a 16 fold downsample in XY and no downsampling in Z
    
    """ If padded earlier, add padding here
    """

    ### Load n5 file
    #with z5py.File(input_name, "r") as f:
    
    f = z5py.File(n5_file, "r")
    dset = f['setup0/timepoint0/s0']    
    

    
    spots_df = pd.read_csv(examples[0]['input'])

    ### parse dataframes
    # first 2 rows are nonsense
    spots_df.drop([0,1,2], inplace=True)
    spots_df = spots_df[['LABEL','ID', 'TRACK_ID', 'POSITION_X', 'POSITION_Y', 'POSITION_Z', 'FRAME', 'MANUAL_SPOT_COLOR']]
    

    
    spots_df = spots_df.reset_index(drop=True)   ### resets index and prevents it from being added as a new column
    # spots_df['FRAME'] = pd.to_numeric(spots_df['FRAME'])
    spots_df['TRACK_ID'] = pd.to_numeric(spots_df['TRACK_ID'])
    spots_df['POSITION_X'] = pd.to_numeric(spots_df['POSITION_X'])
    spots_df['POSITION_Y'] = pd.to_numeric(spots_df['POSITION_Y'])
    spots_df['POSITION_Z'] = pd.to_numeric(spots_df['POSITION_Z'])
    
    
    
    
    spots_df['ID'] = pd.to_numeric(spots_df['ID'])
    
 

    ### If NOT fully tracked dataset, then exclude green color since green is NFO
    if not fully_tracked:
        exclude_ids = np.where(spots_df['MANUAL_SPOT_COLOR'] == 'r=51;g=255;b=51')[0]
    
        spots_df.drop(exclude_ids, inplace=True)
        spots_df = spots_df.reset_index(drop=True)


    
    
    spots_df['cell_type'] = 0   ### by default all other colors and no colors is type mature OL
    
    ### All white color spots have value = 1 (NFO)
    NFO_ids = []
    NFO_ids.append(np.where(spots_df['MANUAL_SPOT_COLOR'] == 'r=255;g=255;b=255')[0])
    spots_df.loc[NFO_ids[0], 'cell_type'] = 1
    
    
    if fully_tracked:   ### then green also means NFO
        NFO_ids = []
        NFO_ids.append(np.where(spots_df['MANUAL_SPOT_COLOR'] == 'r=51;g=255;b=51')[0])
        spots_df.loc[NFO_ids[0], 'cell_type'] = 1
        
        
        WM_ids = []
        WM_ids.append(np.where(spots_df['MANUAL_SPOT_COLOR'] == 'r=204;g=255;b=204')[0])
        spots_df.loc[WM_ids[0], 'cell_type'] = 7



    m_label = np.where(spots_df['LABEL'] == 'm')[0]
    spots_df.loc[m_label, 'cell_type'] = 4


    #############################################################################################
    #%% Save crops
            
    
    crop_size = 40
    z_size = 8
    

    num_corr = 0


    dataset_name = 'MaMuT_corrected'
    
    
    sav_fold = '/media/user/8TB_HDD/LargeOL_training_data/MaMuT_LARGE_CLEANED/'

    for id_r, row in spots_df.iterrows():
        
        
        x = int(row['POSITION_Y'])  ### flipped X and Y
        y = int(row['POSITION_X'])
        z = int(row['POSITION_Z'])
        
        cell_type = row['cell_type']
        
        crop = dset[z-z_size:z+z_size, x-crop_size:x+crop_size, y-crop_size:y+crop_size]
        
        ma = plot_max(crop, plot=0)
        

        tiff.imwrite(sav_fold + filename + '_' + dataset_name + '_CROP_LARGEOL_' + str(id_r) + '_val_' + str(cell_type) + '.tif', crop)
        # tiff.imwrite(sav_dir + filename + '_' + dataset_name + '_MAX_PROJECT_CROP_LARGEOL_' + str(id_r) + '_val_' + str(sorted_df.loc[id_r,'is_OL']) + '.tif', ma)
            
        num_corr += 1
        
        print(id_r)
    
    





