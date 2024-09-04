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
from skimage.filters import threshold_otsu
import json
import matplotlib.pyplot as plt
    
from postprocess_func import *
import seaborn as sns
from scipy import stats   



import sys
sys.path.append("..")



from get_brain_metadata import *



#%%  LOAD UNET

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# from UNet_pytorch import *
# from UNet_pytorch_online import *
# from PYTORCH_dataloader import *
from UNet_functions_PYTORCH import *

import tifffile as tiff

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True

""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


"""  Network Begins: """
s_path = '/media/user/8TB_HDD/Myelin_APOC_seg/Training_blocks/(1) Check_lightsheet_NO_transforms_4deep/'

overlap_percent = 0.5
input_size = 128
depth = 16
num_truth_class = 2

""" TO LOAD OLD CHECKPOINT """
# Read in file names
onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
onlyfiles_check.sort(key = natsort_key1)

""" Find last checkpoint """       
last_file = onlyfiles_check[-1]
split = last_file.split('check_')[-1]
num_check = split.split('.')
checkpoint = num_check[0]
checkpoint = 'check_' + checkpoint
num_check = int(num_check[0])

check = torch.load(s_path + checkpoint, map_location=device)
unet = check['model_type']
unet.load_state_dict(check['model_state_dict'])
unet.eval()
#unet.training # check if mode set correctly
unet.to(device)

print('parameters:', sum(param.numel() for param in unet.parameters()))

input_path = '/media/user/8TB_HDD/Myelin_APOC_seg/Training_blocks/Training_blocks_blocks_128_16_UNet/'
mean_arr = np.load(input_path + 'normalize/mean_VERIFIED.npy')
std_arr = np.load(input_path + 'normalize/std_VERIFIED.npy')       




#%% List of brains
list_brains = get_metadata(mouse_num = ['M260'])

# list_brains = get_metadata(mouse_num = 'all')


ANTS = 1


#%% Parse the json file so we can choose what we want to extract or mask out

#reference_atlas = '/home/user/.brainglobe/633_princeton_mouse_20um_v1.0/annotation.tiff'

# reference_atlas = '/home/user/.brainglobe/633_perens_lsfm_mouse_20um_v1.0/annotation.tiff'

reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation.tiff'




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



#%% Loop through each experiment
for exp in list_brains:
    keys_df = main_keys.copy(deep=True)

    input_path = exp['path']
    name_str = exp['name']
    
    n5_file = input_path + name_str + '.n5'
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'

    pad = True
    
    XY_res = 1.152035240378141
    Z_res = 5
    
    
    res_diff = XY_res/Z_res
    
    ### Initiate poolThread
    #poolThread_load_data = ThreadPool(processes=1)
    #poolThread_post_process = ThreadPool(processes=2)
    
    """ Loop through all the folders and do the analysis!!!"""
    #filename = n5_file.split('/')[-2]
    filename = n5_file.split('/')[-1].split('.')[0:-1]
    filename = '.'.join(filename) 
    sav_dir = input_path + '/' + filename + '_postprocess'
    
    analysis_dir = input_path + name_str + '_MaskRCNN_patches/'
    """ For testing ILASTIK images """
    images = glob.glob(os.path.join(analysis_dir,'*_df.pkl'))    # can switch this to "*truth.tif" if there is no name for "input"
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
    

    #%% Get atlas for overlay

    #%% Parse the json file so we can choose what we want to extract or mask out
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
    
    
    input_path = exp['path']
    name_str = exp['name']
    
    n5_file = input_path + name_str + '.n5'
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
    #atlas_dir = downsampled_dir + name_str + '633_princeton_mouse_20u
    
    # actual everything, no gauss, -15 GRID
    # atlas_dir = downsampled_dir + name_str + '633_perens_lsfm_mouse_20um_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n40.9_n4_1_grid_-15_gauss_0_use_steps_default_PADDED_50/'
    
    allen_dir = downsampled_dir + name_str + '_ISOCORTEX_CORTEX_ONLY_allen_mouse_10um_bend_0.95_grid_-10_gauss_0/'
    analysis_dir = input_path + name_str + '_MaskRCNN_patches/'
    
    
    ### NEW WITH ALLEN
    if exp['thresh'] != 0: ### for DIVIDED BY MYELIN
    
        ### WHOLE BRAIN registered to Allen --- use this for cerebellum!!!
        # atlas_dir = downsampled_dir + name_str + 'allen_mouse_20um_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-15_gauss_0_use_steps_default_PADDED_50/'
        atlas_WHOLE_dir = downsampled_dir + name_str + 'allen_mouse_20um_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default_PADDED_50/'
    
        ### This is just cortex
        if ANTS:
            ### cortex registered to Allen
            # atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str + 'allen_mouse_CORTEX_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
       
            ## cortex registered using MYELIN brain average template
            atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'

            # ### cortex registered using OUR OWN CUBIC AUTOFLUORESCENCE
            # atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_20um_CUBIC_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'


    else:
        
        ### NEED A ATLAS_WHOLE_DIR for CUBIC brain!!! i.e. whole CUBIC template brain, including cerebellum
        # atlas_dir = downsampled_dir + name_str + 'allen_mouse_20um_CUBIC_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-15_gauss_0_use_steps_default_PADDED_50/'
        atlas_WHOLE_dir = downsampled_dir + name_str + 'allen_mouse_20um_CUBIC_FULLBRAIN_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default_PADDED_50/'
           
        
        ### This is just cortex
        if ANTS:
            atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_20um_CUBIC_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
            print('CUBIC reference')
    
    # myelin_path = glob.glob(os.path.join(downsampled_dir,'*_ch0_n4_down1_PAD.tif'))[0]    # can switch this to "*truth.tif" if there is no name for "input"
      
    myelin_path = glob.glob(os.path.join(downsampled_dir,'*_ch0_n4_down1_resolution_20_PAD.tif'))[0]    # can switch this to "*truth.tif" if there is no name for "input"
   
    
    n5_file = input_path + name_str + '.n5'
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
    
    
    
    #%% Get registered atlas and shift the axes
    myelin = tiff.imread(myelin_path)
    
    

    
    """ USE THE FULL ATLAS --- combined cerebellum/IC + Cortex!!! """
    atlas = tiff.imread(atlas_dir + '/atlas_combined_CEREBELLUM_AND_IC.tif')
    
    # zzz
    # atlas = tiff.imread(atlas_dir + '/registered_atlas.tiff')        
    
    
    atlas = np.moveaxis(atlas, 0, 1)   ### reshuffle atlas so now in proper orientation
    atlas = np.flip(atlas, axis=0)  ### flip the Z-axis
    atlas = np.flip(atlas, axis=2)
    atlas_size_pre_resize = atlas.shape
    atlas = resize(atlas, myelin.shape, anti_aliasing=False, order=0, preserve_range=True)   ### rescale the images
        
    """ If padded earlier, REMOVE padding here
    """
    
    ### Load n5 file
    #with z5py.File(input_name, "r") as f:
    
    f = z5py.File(n5_file, "r")
    dset = f['setup0/timepoint0/s0']    
     
    
    ### OPTIONALLY IF DONT WANT TO LOAD DSET FROM N5 file
    reg_name = np.load(atlas_dir + 'brainreg_args.npy')
    z_after = int(reg_name[-1])
    x_after = int(reg_name[-2])
    
    # find actual down_factor while considering the padding on each side of the image
    m_shape = np.asarray(myelin.shape)
    m_shape[0] = m_shape[0] - z_after*2
    m_shape[1] = m_shape[1] - x_after*2
    m_shape[2] = m_shape[2] - x_after*2
    
    down_factor = dset.shape/m_shape
        
    ### REMOVE PADDING
    myelin_nopad = myelin[z_after:-z_after, x_after:-x_after, x_after:-x_after]
    atlas_nopad = atlas[z_after:-z_after, x_after:-x_after, x_after:-x_after]



    #%% LOAD PICKLE SO DONT HAVE TO REDO THIS CONSTANTLY IN THE FUTURE
    if os.path.isfile(sav_dir + filename + '_ALL_CONCAT_MYELIN_EXTRACTION.pkl'):
        all_concat = pd.read_pickle(sav_dir + filename + '_ALL_CONCAT_MYELIN_EXTRACTION.pkl')
        
        print('LOADING PREVIOUS SAVE')
        
        zzz
    else:
    
        f = z5py.File(n5_file, "r")
        
        dset = f['setup0/timepoint0/s0']
        
            
    
        all_blocks_df_intensity = [] 
        # for blk_num in range(0, len(examples)):
            
        for blk_num in range(0, len(examples)):
         
            #blk_num = 100
            #blk_num = 200
            pkl = pd.read_pickle(examples[blk_num]['pkl'])
            
            if len(pkl) == 0:
                print('No cells in pickle?')
                continue
            
            
            def get_im(dset, s_c, Lpatch_depth, Lpatch_size):
                
                    #tic = time.perf_counter()
                    
                    ### If nearing borders of image, prevent going out of bounds!
                    z_top = s_c[2] + Lpatch_depth
                    if z_top >= dset.shape[0]: z_top = dset.shape[0]
                    
                    y_top = s_c[1] + Lpatch_size
                    if y_top >= dset.shape[1]: y_top = dset.shape[1]
                    
                    x_top = s_c[0] + Lpatch_size
                    if x_top >= dset.shape[2]: x_top = dset.shape[2]
                    
                    input_im = dset[s_c[2]:z_top, s_c[1]:y_top, s_c[0]:x_top]
                    og_shape = input_im.shape
                    
                    #toc = time.perf_counter()
                    # print('loaded asynchronously')
                    
                    #print(f"Opened subblock in {toc - tic:0.4f} seconds")
                    
                    return input_im, og_shape            
        
    
            s_c = pkl['xyz_offset'][0]
    
            Lpatch_size = 128 * 10
            Lpatch_depth = 64 * 4
            input_im, og_shape = get_im(dset, s_c, Lpatch_depth, Lpatch_size)
            
            
            ### can be nan at end of image?
            input_im[input_im == np.nan] = 0
            
            
            #%% Convert detected cells to array
            coords = pkl['coords_raw']
            overlap_pxy = pkl['overlap_pxy'][0]
            overlap_pz = pkl['overlap_pz'][0]
            
            # all_coords = np.vstack(coords)
            # all_coords[:, 0] = all_coords[:, 0] - overlap_pz
            # all_coords[:, 0][all_coords[:, 0]  >= np.shape(input_im)[0]] = np.shape(input_im)[0] - 1
            
            
            # all_coords[:, 1] = all_coords[:, 1] - overlap_pxy
            # all_coords[:, 1][all_coords[:, 1]  >= np.shape(input_im)[1]] = np.shape(input_im)[1] - 1
            
            # all_coords[:, 2] = all_coords[:, 2] - overlap_pxy
            # all_coords[:, 2][all_coords[:, 2]  >= np.shape(input_im)[2]] = np.shape(input_im)[2] - 1
     
                        
            all_detections = np.zeros(np.shape(input_im))
            ### Must loop through and go one by one
            for num, c in enumerate(coords):
                #all_coords = np.vstack(coords)
                c[:, 0] = c[:, 0] - overlap_pz
                c[:, 0][c[:, 0]  >= np.shape(input_im)[0]] = np.shape(input_im)[0] - 1
                
                
                c[:, 1] = c[:, 1] - overlap_pxy
                c[:, 1][c[:, 1]  >= np.shape(input_im)[1]] = np.shape(input_im)[1] - 1
                
                c[:, 2] = c[:, 2] - overlap_pxy
                c[:, 2][c[:, 2]  >= np.shape(input_im)[2]] = np.shape(input_im)[2] - 1
                        
                all_detections[c[:, 0], c[:, 1], c[:, 2]] = num + 1
                
            all_detections = np.asarray(all_detections, dtype=np.uint32)
                
            bw_detections = np.copy(all_detections)
            bw_detections[bw_detections > 0] = 1
            
            
    
            
            #%% Extract myelin by removing OL cell bodies first                
            # ### DILATE
            from skimage.morphology import binary_dilation, ball
            ball = ball(radius = 2)
            dil_im = binary_dilation(bw_detections, footprint=ball)
            
    
    
            #%% CROP ATLAS SO CAN INSERT DENSITY MEASURES
    
    
            ### get atlas crop
            atlas_crop, og_shape = get_im(atlas_nopad, np.asarray(s_c/np.roll(down_factor, 2), dtype=np.int32),
                                          np.asarray(Lpatch_depth/np.roll(down_factor, 2)[-1], dtype=np.int32), 
                                          np.asarray(Lpatch_size/np.roll(down_factor, 2)[0], dtype=np.int32))
            
            
            
            atlas_upsampled = resize(atlas_crop, input_im.shape, anti_aliasing=False, order=0, preserve_range=True)
            
            
            
            #%% Create boundary image for reference atlas
            
            if blk_num == 226 or blk_num == 227:
                from skimage.segmentation import find_boundaries
                bound_slice = np.zeros(np.shape(atlas_upsampled))
                for slice_num, slice_im in enumerate(atlas_upsampled):
                    bw_slice = find_boundaries(slice_im)
                    bound_slice[slice_num, :, :] = bw_slice
        
                bound_slice = np.asarray(bound_slice, dtype=np.uint8)
        
                    
                tiff.imwrite(sav_dir + filename + '_' + str(int(pkl['block_num'][0])) + '_' + str(blk_num) + '_dilated_cell_bodies.tif', np.asarray(dil_im, dtype=np.uint8))
                tiff.imwrite(sav_dir + filename + '_' + str(int(pkl['block_num'][0])) + '_' + str(blk_num) + '_atlas_upsampled.tif', np.asarray(atlas_upsampled, dtype=np.uint32))
        
                tiff.imwrite(sav_dir + filename + '_' + str(int(pkl['block_num'][0])) + '_' + str(blk_num) + '_atlas_bounds.tif', np.asarray(bound_slice, dtype=np.uint8))
        
    
    
    
            #%% Run UNet for myelin seg
                    
            print('Running UNet segmentation')
            from UNet_functions_PYTORCH import *
            
            segmentation = UNet_inference_by_subparts_PYTORCH(unet, device, input_im, overlap_percent, quad_size=input_size, quad_depth=depth,
                                                      mean_arr=mean_arr, std_arr=std_arr, num_truth_class=num_truth_class,
                                                      skip_top=1)
            
            segmentation[segmentation > 0] = 1
            
            seg_masked = np.copy(segmentation)
            seg_masked[dil_im > 0] = 0    ### set as ZERO so can ignore later
            
                    
            ### Mask out input_im
            masked = np.copy(input_im)
            masked[seg_masked == 0] = 0 
            
            
            # import napari
            # viewer = napari.Viewer()
            
            # new_layer = viewer.add_image(atlas_nopad)       
    
            # new_layer = viewer.add_image(myelin_nopad)           
            
            # new_layer = viewer.add_image(atlas_upsampled)       
    
            # new_layer = viewer.add_image(masked)       
    
            # new_layer = viewer.add_image(segmentation)       
            # # zzz
    
        
            # viewer.show(block=True)
                    
            
            
            
            #%% Figure out each associated region in atlas and find density
            
            print('Extracting intensity densities and areas')
            ### go through CC and accumulate
            cc = regionprops(atlas_upsampled)
            
            
            df_all_regions = pd.DataFrame()
            
            for region in cc:
                
                coords = region['coords']
                
                id_region = atlas_upsampled[coords[:, 0], coords[:, 1], coords[:, 2]][0]
                
                
                vals = masked[coords[:, 0], coords[:, 1], coords[:, 2]]
                
                vals = vals[np.where(vals != 0)[0]]   ### IGNORE ZEROS for calculations
                
                
                
                ### also find out how many voxels belong to cell body so can exclude them from final density calculations!
                val_cell_bodies = dil_im[coords[:, 0], coords[:, 1], coords[:, 2]]
                val_cell_bodies = val_cell_bodies[np.where(val_cell_bodies > 0)[0]]
                
    
                dict_reg = {'reg_id':id_region, 'reg_voxels':len(coords), 'cell_body_voxels':len(val_cell_bodies),
                            'sum_int':np.sum(vals), 'num_vals':len(vals), 'mean':np.sum(vals)/len(vals)}
                
                df_all_regions = pd.concat([df_all_regions, pd.DataFrame([dict_reg])], ignore_index=True)
    
            all_blocks_df_intensity.append(df_all_regions)
            
            
            # tiff.imwrite(sav_dir + filename + '_' + str(int(pkl['block_num'][0])) + '_' + str(blk_num) + '_input_im_nomask.tif', np.asarray(input_im, dtype=np.uint16))
            # tiff.imwrite(sav_dir + filename + '_' + str(int(pkl['block_num'][0])) + '_' + str(blk_num) + '_segmentation.tif', np.asarray(segmentation, dtype=np.uint8))
            
            print(blk_num)
            
            
        
        all_concat = pd.concat(all_blocks_df_intensity)
        all_concat.reset_index(inplace=True, drop=True)
        
        #%% SAVE PICKLE SO DONT HAVE TO REDO THIS CONSTANTLY IN THE FUTURE
        all_concat.to_pickle(sav_dir + filename + '_ALL_CONCAT_MYELIN_EXTRACTION.pkl')
    
    


    
    
    """ CODE BELOW USED TO ACTUALLY GENERATE FIGURES --- should put this into seperate post-processing file """

    ### Sort through keys_df and pool by id region!
    keys_df['scaled_intden'] = np.nan
    for id_r, reg_row in keys_df.iterrows():
        
        reg = reg_row['ids']
        
        
        match_blks = np.where(all_concat['reg_id'] == reg)[0]
        
        
        if len(match_blks) == 0:
            continue
        
        
        ### First sum altogether and get mean intensity of all voxels
        mean_intensity_over_myelin_density = np.sum(all_concat.iloc[match_blks]['sum_int'])/np.sum(all_concat.iloc[match_blks]['num_vals'])
        
        ### Then multiply mean by area --- and then divide by TOTAL region of brain to get scaled A.U. intden
        ### ****************** BUT ALSO NEED TO SUBTRACT THE VOLUME OF THE OL SOMAS!
        
        
        scaled_intden = (mean_intensity_over_myelin_density * np.sum(all_concat.iloc[match_blks]['num_vals'])) / (np.sum(all_concat.iloc[match_blks]['reg_voxels']) - np.sum(all_concat.iloc[match_blks]['cell_body_voxels']))
        
        
        keys_df.loc[id_r, 'scaled_intden'] = scaled_intden
        
        keys_df.loc[id_r, 'mean_volume_myelin'] = np.sum(all_concat.iloc[match_blks]['num_vals']) / np.sum(all_concat.iloc[match_blks]['reg_voxels'])



    #%% Next pool together different sub-regions to make overarching regions
    # df_level = keys_df.sort_values(by=['st_level'], ascending=False, ignore_index=True)
    
    # for i, row in df_level.iterrows():
        
    #     childs = row['children']
        
    #     if len(childs) > 0 and np.isnan(row['atlas_vol_W']):  ### if current row is NAN, then want to pool children, otherwise no
                    
    #         df_childs = pd.DataFrame()
    #         for child in childs:
                
    #             id_c = np.where(df_level['ids'] == child)[0][0]
                
    #             child_row = df_level.iloc[id_c]
                
    #             child_df = pd.DataFrame([child_row])
                
    #             df_childs = pd.concat([df_childs, child_df], axis=0)
                
    #         df_sum = df_childs.sum(axis=0, numeric_only=True)
            
    #         df_sum = df_sum.drop(['ids', 'parent', 'st_level'])
            
    #         row[df_sum.index] = df_sum
            
            
    #         df_level.iloc[i] = row  ### add row back in
          
    # keys_df_MYELIN = df_level
    
    keys_df_MYELIN = keys_df


    
    #%% List of brains
    list_brains = get_metadata(mouse_num = ['M260'])
 
    sav_fold = sav_dir
    

    pad = True
    
    XY_res = 1.152035240378141
    Z_res = 5
    res_diff = XY_res/Z_res
    
    #%%%% Parse the json file so we can choose what we want to extract or mask out
    with open('../atlas_ids/atlas_ids.json') as json_file:
        data = json.load(json_file)
     
         
    data = data['msg'][0]
    
    keys_tmp = get_ids_all(data, all_keys=[], keywords=[''])  
    keys_tmp = pd.DataFrame.from_dict(keys_tmp)
    
    
    #%%%% Parse pickles from all folders
    """ Loop through all the folders and pool the dataframes!!!"""
    
    all_coords_df = []
    all_keys_df = []
    for id_f, info in enumerate(list_brains):
    
        fold = info['path'] + info['name'] + '_postprocess'
    
        coords = glob.glob(os.path.join(fold,'*_coords_df_ALLEN_EVERYTHING-15grid.pkl'))    # can switch this to "*truth.tif" if there is no name for "input"
        keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-15grid.pkl'))    
        
        keys_df = pd.read_pickle(keys[0])
        coords_df = pd.read_pickle(coords[0])
        
        print('number of cells: ' + str(len(coords_df)))
        
        keys_df['acronym'] = keys_tmp['acronym']
        keys_df['dataset'] = info['num']
        keys_df['exp'] = info['exp']
        keys_df['sex'] = info['sex']
        keys_df['age'] = info['age']   
    
        
        # #%%%% sort by depth level and pool all values from children
        
        # df_level = keys_df.sort_values(by=['st_level'], ascending=False, ignore_index=True)
        
        # for i, row in df_level.iterrows():
            
        #     childs = row['children']
            
        #     if len(childs) > 0 and np.isnan(row['atlas_vol_W']):  ### if current row is NAN, then want to pool children, otherwise no
                        
        #         df_childs = pd.DataFrame()
        #         for child in childs:
                    
        #             id_c = np.where(df_level['ids'] == child)[0][0]
                    
        #             child_row = df_level.iloc[id_c]
                    
        #             child_df = pd.DataFrame([child_row])
                    
        #             df_childs = pd.concat([df_childs, child_df], axis=0)
                    
        #         df_sum = df_childs.sum(axis=0, numeric_only=True)
                
        #         df_sum = df_sum.drop(['ids', 'parent', 'st_level', 'density_W', 'age'])
                
        #         row[df_sum.index] = df_sum
                
                
        #         df_level.iloc[i] = row  ### add row back in
              
        # keys_df = df_level
        
        if 'side' in info:
            
            keys_df['density_W'] = keys_df['num_OLs_' + info['side']]/keys_df['atlas_vol_' + info['side']]
            keys_df['density_LARGE_W'] = keys_df['num_large_' + info['side']]/keys_df['atlas_vol_' + info['side']] 
            
        else:
            #keys_df['density_L'] = keys_df['num_OLs_L']/keys_df['atlas_vol_L']
            #keys_df['density_R'] = keys_df['num_OLs_R']/keys_df['atlas_vol_R']
            keys_df['density_W'] = keys_df['num_OLs_W']/keys_df['atlas_vol_W']
            keys_df['density_LARGE_W'] = keys_df['num_large_W']/keys_df['atlas_vol_W']
    
    
    #%% And then read in the M260 keys_df OL counts to compare!
    keys_df['myelin_intden'] = keys_df_MYELIN['scaled_intden']
    
    keys_df['mean_volume_myelin'] = keys_df_MYELIN['mean_volume_myelin']



    ### Makes it so that svg exports text as editable text!
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none' 
    
    #%% Plot
    sav_fold = '/media/user/8TB_HDD/Plot_outputs/'
    fontsize = 14


    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    pl = sns.regplot(x='density_W', y='myelin_intden', data=keys_df.dropna(), scatter_kws={"color": "grey", 's':10}, line_kws={"color": "red", "alpha":0.2}, ax=ax)
    
    #calculate slope and intercept of regression equation
    slope, intercept, r, p, sterr = stats.linregress(x=pl.get_lines()[0].get_xdata(),
                                                           y=pl.get_lines()[0].get_ydata())
    #display slope and intercept of regression equation
    print(slope)
    
    r,p = stats.pearsonr(keys_df.dropna()['density_W'], keys_df.dropna()['myelin_intden'])
    print(r)
    # print(p)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    
    
    ### Force calculate p value
    from mpmath import mp
    # mp.dps = 1000
    
    r = mp.mpf(r)
    n = len(keys_df.dropna()['density_W'])
    
    x = (-abs(r) + 1)/2  # shift per `loc=-1`, scale per `scale=2`
    p = 2*mp.betainc(n/2 - 1, n/2 - 1, 0, x, regularized=True)
    print(p)



    # plt.xlim([0, 300000])
    plt.ylim([0, 600])
    ax.ticklabel_format(axis='x', scilimits=[-3, 3])  ### set to be order of magnitude
    ax.ticklabel_format(axis='y', scilimits=[-3, 3])
    
    plt.xlabel('Density of OLs (cells/mm\u00b3)', fontsize=fontsize)
    plt.ylabel('Intensity density myelin', fontsize=fontsize)
    plt.tight_layout()
    
    plt.savefig(sav_fold + 'COMPARE_MYELIN_intensity_density_vs_OL_density_Rval_' + '.png', format='png', dpi=300)
    plt.savefig(sav_fold + 'COMPARE_MYELIN_intensity_density_vs_OL_density_Rval_' + '.svg', format='svg', dpi=300)
    
    
    
    ### plot volume vs. density
    
    
    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    pl = sns.regplot(x='density_W', y='mean_volume_myelin', data=keys_df.dropna(), scatter_kws={"color": "grey", 's':10}, line_kws={"color": "red", "alpha":0.2}, ax=ax)
    
    # pl = sns.regplot(x='density_R', y='density_L', data=mean_hemisphere.dropna(), scatter_kws={"color": "grey", 's':10}, line_kws={"color": "red", "alpha":0.2}, ax=ax)
    
    #calculate slope and intercept of regression equation
    slope, intercept, r, p, sterr = stats.linregress(x=pl.get_lines()[0].get_xdata(),
                                                           y=pl.get_lines()[0].get_ydata())
    
    #display slope and intercept of regression equation
    print(slope)
    # print(p)



    
    r,p = stats.pearsonr(keys_df.dropna()['density_W'], keys_df.dropna()['mean_volume_myelin'])
    print(r)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ### Force calculate p value
    from mpmath import mp
    # mp.dps = 1000
    
    r = mp.mpf(r)
    n = len(keys_df.dropna()['density_W'])
    
    x = (-abs(r) + 1)/2  # shift per `loc=-1`, scale per `scale=2`
    p = 2*mp.betainc(n/2 - 1, n/2 - 1, 0, x, regularized=True)
    print(p)

    
    
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)

    # plt.xlim([0, 300000])
    plt.ylim([0, 1])
    ax.ticklabel_format(axis='x', scilimits=[-3, 3])  ### set to be order of magnitude
    ax.ticklabel_format(axis='y', scilimits=[-3, 3])
    
    plt.xlabel('Density of OLs (cells/mm\u00b3)', fontsize=fontsize)
    plt.ylabel('Myelin volume / region volume', fontsize=fontsize)
    plt.tight_layout()
    
    plt.savefig(sav_fold + 'COMPARE_MYELIN_VOLUME_vs_OL_density_Rval_'  + '.png', format='png', dpi=300)
    plt.savefig(sav_fold + 'COMPARE_MYELIN_VOLUME_vs_OL_density_Rval_'  + '.svg', format='svg', dpi=300)
    
    






    
