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

### Makes it so that svg exports text as editable text!
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'    



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






def df_to_im(plot_df, atlas):
    cell_pos = [plot_df['Z_down'].astype(int), plot_df['X_down'].astype(int), plot_df['Y_down'].astype(int)]
    cell_pos = np.transpose(np.asarray(cell_pos))
    
    ### make sure coordinates stay in range
    cell_pos[np.where(cell_pos[:, 0] >= atlas.shape[0])[0], 0] = atlas.shape[0] - 1
    cell_pos[np.where(cell_pos[:, 1] >= atlas.shape[1])[0], 1] = atlas.shape[1] - 1
    cell_pos[np.where(cell_pos[:, 2] >= atlas.shape[2])[0], 2] = atlas.shape[2] - 1 
    
    print('saving cells image')
    cells_im = np.zeros(np.shape(atlas))
    cells_im[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]] = 255  


    return cells_im, cell_pos

        
        
"""  Network Begins: """


# s_path = '/media/user/8TB_HDD/LargeOL_training_data/(2) Check_lightsheet_NO_transforms_3deep_LARGEOL_batchnorm_kernel3_slow1e-6/'


# s_path = '/media/user/8TB_HDD/LargeOL_training_data/(5) Check_lightsheet_NO_transforms_4deep_LARGEOL_NO_batchnorm_kernel3_lr1e-5_INCLUDE_week3/'

# s_path = '/media/user/8TB_HDD/LargeOL_training_data/(8) Check_lightsheet_NO_transforms_3deep_LARGEOL_batchnorm_kernel3_slow1e-6_NOweek3/'


# s_path = '/media/user/8TB_HDD/LargeOL_training_data/(11) Check_lightsheet_NO_transforms_2deep_LARGEOL_YES_batchnorm_kernel3_slow1e-5_NOweek3_MORE_DATA_wf6/'


### This was working okay! - 1st runs
# s_path = '/media/user/8TB_HDD/LargeOL_training_data/(12) Check_lightsheet_NO_transforms_3deep_LARGEOL_YES_batchnorm_kernel3_slow1e-5_NOweek3_MORE_DATA_wf5/'


# s_path = '/media/user/8TB_HDD/LargeOL_training_data/(16) Check_lightsheet_NO_transforms_3deep_LARGEOL_YES_batchnorm_kernel3_ADAPT_LR_NO_week3_MORE_DATA_AGING_wf5/'


# s_path = '/media/user/8TB_HDD/LargeOL_training_data/(17) Check_lightsheet_NO_transforms_3deep_LARGEOL_YES_batchnorm_kernel3_ADAPT_LR_NO_week3_wf5_FULLY_NEW_MAMUT/'
   

# s_path = '/media/user/8TB_HDD/LargeOL_training_data/(18) Check_lightsheet_NO_transforms_3deep_LARGEOL_YES_batchnorm_kernel3_ADAPT_LR_NO_week3_wf5_FULLY_NEW_MAMUT_include6/'
   
        
s_path = '/media/user/8TB_HDD/LargeOL_training_data/(20) Check_lightsheet_NO_transforms_3deep_LARGEOL_YES_batchnorm_kernel3_ADAPT_LR_NO_week3_wf5_FULLY_NEW_MAMUT_CLEANED_batch8/'
           
            
           
    
# input_path = '/media/user/8TB_HDD/LargeOL_training_data/LargeOL_training_data_blocks_80_16_UNet/'


# input_path = '/media/user/8TB_HDD/LargeOL_training_data/FULL_TRAINING_SET_NEW_include67/FULL_TRAINING_SET_NEW_include67_blocks_80_16_UNet/'

input_path = '/media/user/8TB_HDD/LargeOL_training_data/FULL_TRAINING_SET_NEW_CLEANED/FULL_TRAINING_SET_NEW_CLEANED_blocks_80_16_UNet/'



overlap_percent = 0.5
input_size = 80
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





mean_arr = np.load(input_path + 'normalize/mean_VERIFIED.npy')
std_arr = np.load(input_path + 'normalize/std_VERIFIED.npy')       








### Ran these during testing phase

# list_brains = get_metadata(mouse_num = ['M254'])   # ran?
# list_brains = get_metadata(mouse_num = ['M260'])   ### ran
# list_brains = get_metadata(mouse_num = ['M286'])  ### ran, used bottom thresh of 200 instead of 250 for large_df

# list_brains = get_metadata(mouse_num = ['M312'])   ### ran

# list_brains = get_metadata(mouse_num = ['M265'])   ### ran




#%% List of brains to analyze
# list_brains = get_metadata(mouse_num = ['M256', 'M260','M127', 'M126', 'M254', 'M299'])   ### P60, cup, and Recovery
# exclude M229 due to translation only stitching!!!   #also 'M223'


# run as 3 separate instances
# list_brains = get_metadata(mouse_num = ['M127', 'M126'])   ### P60, cup, and Recovery
# list_brains = get_metadata(mouse_num = ['M256', 'M260'])#'M127', 'M126', 'M254', 'M299'])   ### P60, cup, and Recovery
# list_brains = get_metadata(mouse_num = ['M254', 'M299'])   ### P60, cup, and Recovery



# 8 mos brains
# list_brains = get_metadata(mouse_num = ['M279', 'M286', 'M281', 'M285'])

# 22 mos brains
# list_brains = get_metadata(mouse_num = ['M271', 'M91', 'M97', 'M334']) (current)

# 30 mos brains
# list_brains = get_metadata(mouse_num = ['1Otx7', 'Otx18', '5Otx5', 'Otx6']) (current)



# Cuprizone
# list_brains = get_metadata(mouse_num = ['M265', 'M266', 'M267'])

# Recovery 
# list_brains = get_metadata(mouse_num = ['M312', 'M310', 'M313'])  # (current)



# list_brains = get_metadata(mouse_num = ['M147', 'M170'])    ### FVB, CD1

# list_brains = get_metadata(mouse_num = ['M246'])  ### 96rik for comparison

# list_brains = get_metadata(mouse_num = ['5Otx5'])     ### correcting with MaMut (Fully done)

# list_brains = get_metadata(mouse_num = ['M312'])     ### correcting with MaMut

# list_brains = get_metadata(mouse_num = ['M265'])     ### correcting with MaMut


# list_brains = get_metadata(mouse_num = ['M271'])     ### correcting with MaMut  --- DID NOT CORRECT
# list_brains = get_metadata(mouse_num = ['M334'])     ### correcting with MaMut

# list_brains = get_metadata(mouse_num = ['M91'])     ### correcting with MaMut (Fully done)


# list_brains = get_metadata(mouse_num = ['Otx6'])     ### correcting with MaMut
# list_brains = get_metadata(mouse_num = ['Otx18'])     ### correcting with MaMut
# list_brains = get_metadata(mouse_num = ['M281'])     ### correcting with MaMut


# list_brains = get_metadata(mouse_num = ['M256'])     ### correcting with MaMut

# list_brains = get_metadata(mouse_num = ['M127'])     ### correcting with MaMut


######################## REDO analysis with newly trained ViT
# list_brains = get_metadata(mouse_num = ['M256'])     ### correcting with MaMut
# list_brains = get_metadata(mouse_num = ['5Otx5']) 
# list_brains = get_metadata(mouse_num = ['M281', 'M286'])     ### correcting with MaMut




##############NOT DONE
# run as 3 separate instances
# list_brains = get_metadata(mouse_num = [])   ### P60, cup, and Recovery
# list_brains = get_metadata(mouse_num = ['M299', 'M256', 'M260', 'M127', 'M126'])#'M127', 'M126', 'M254', 'M299'])   ### P60, cup, and Recovery
### DONE: 'M254'

# 8 mos brains
# list_brains = get_metadata(mouse_num = ['M285', 'M279'])   ### DONE

# 22 mos brains
# list_brains = get_metadata(mouse_num = ['M334', 'M97', 'M91']) #(current)   ### DONE: 'M271'

# 30 mos brains
# list_brains = get_metadata(mouse_num = ['1Otx7', 'Otx18', 'Otx6']) #(current)




# Cuprizone
# list_brains = get_metadata(mouse_num = ['M265', 'M266', 'M267'])

# Recovery 
# list_brains = get_metadata(mouse_num = ['M312', 'M310', 'M313'])  # (current)







list_brains = get_metadata(mouse_num = ['M299', 'M256', 'M254', 'M260', 'M127', 'M126',
                                        'M285', 'M279', 'M281', 'M286',
                                        'M334',  'M97',  'M91', 'M271',
                                        '1Otx7', 'Otx18', '5Otx5', 'Otx6',
                                        'M265', 'M266', 'M267',
                                        'M312', 'M310', 'M313'])  # (current)






run_name = '_THRESH200_2_3pool_NEWCNN_CLEANED'
# run_name = '_THRESH200_2_3pool_NEWViT'

# run_name = '_LOWTHRESH'

# run_name = ''
cloudreg = 0

ANTS = 1

CNN = 0

#%% Parse the json file so we can choose what we want to extract or mask out

#reference_atlas = '/home/user/.brainglobe/633_princeton_mouse_20um_v1.0/annotation.tiff'

# reference_atlas = '/home/user/.brainglobe/633_perens_lsfm_mouse_20um_v1.0/annotation.tiff'

reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation.tiff'



ref_atlas = tiff.imread(reference_atlas)
ref_atlas = np.asarray(ref_atlas, dtype=np.uint32)





#%% Keys
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

    ### NEW WITH ALLEN
    if exp['thresh'] != 0: ### for DIVIDED BY MYELIN
    
        ### WHOLE BRAIN registered to Allen --- use this for cerebellum!!!
        # atlas_dir = downsampled_dir + name_str + 'allen_mouse_20um_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-15_gauss_0_use_steps_default_PADDED_50/'
        atlas_WHOLE_dir = downsampled_dir + name_str + 'allen_mouse_20um_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default_PADDED_50/'
    
        ### This is just cortex
        if ANTS:
            ### cortex registered to Allen
            # atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str + 'allen_mouse_CORTEX_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
       
            ### cortex registered using MYELIN brain average template
            atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'

            ### cortex registered using OUR OWN CUBIC AUTOFLUORESCENCE
            # atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_20um_CUBIC_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'


    else:
        
        ### NEED A ATLAS_WHOLE_DIR for CUBIC brain!!! i.e. whole CUBIC template brain, including cerebellum
        # atlas_dir = downsampled_dir + name_str + 'allen_mouse_20um_CUBIC_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-15_gauss_0_use_steps_default_PADDED_50/'
        atlas_WHOLE_dir = downsampled_dir + name_str + 'allen_mouse_20um_CUBIC_FULLBRAIN_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default_PADDED_50/'
           
        
        ### This is just cortex
        if ANTS:
            atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_20um_CUBIC_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
            print('CUBIC reference')
    scaled = 1
    
    
    

    allen_dir = downsampled_dir + name_str + '_ISOCORTEX_CORTEX_ONLY_allen_mouse_10um_bend_0.95_grid_-10_gauss_0/'
    analysis_dir = input_path + name_str + '_MaskRCNN_patches/'
    

    myelin_path = glob.glob(os.path.join(downsampled_dir,'*_ch0_n4_down1_resolution_20_PAD.tif'))[0]    # can switch this to "*truth.tif" if there is no name for "input"
    auto_path = glob.glob(os.path.join(downsampled_dir, '*_ch1_n4_down1_resolution_20_PAD.tif'))[0]
    
 
    pad = True
    
    XY_res = 1.152035240378141
    Z_res = 5
    
    
    res_diff = XY_res/Z_res

    """ Loop through all the folders and do the analysis!!!"""
    filename = n5_file.split('/')[-1].split('.')[0:-1]
    filename = '.'.join(filename) 
    sav_dir = input_path + '/' + filename + '_postprocess'
    
    
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
    

    
    ### Just re-open coords tmp from postprocess RegRCNN
    coords_df = pd.read_pickle(sav_dir + filename + '_coords_TMP.pkl')
    pkl = pd.read_pickle(examples[100]['pkl'])
    overlap_pxy = pkl['overlap_pxy'][0]
    overlap_pz = pkl['overlap_pz'][0]
    

    
    
    

    #%% Load
    myelin = tiff.imread(myelin_path)
              
      
    
    #%% Only do gray matter comparisons to start
    print('Converting coordinates to correct scale')
    ### supposedly atlass is at a 16 fold downsample in XY and no downsampling in Z
    
    """ If padded earlier, add padding here
    """

    ### Load n5 file
    #with z5py.File(input_name, "r") as f:
    
    f = z5py.File(n5_file, "r")
    dset = f['setup0/timepoint0/s0']    
     
    

    # #%% Figure out region that is merged and set different volume threshold in those areas for counting large cells
    
    autofluor = f['setup1/timepoint0/s3']    
    
    down_XY = dset.shape[1]/autofluor.shape[1]
    down_Z = dset.shape[0]/autofluor.shape[0]
    
            
    resolution = 20   
    
    XY_scale = (XY_res * down_XY)/resolution
    Z_scale = (Z_res * down_Z)/resolution
        
    autofluor = np.asarray(autofluor, dtype=np.uint16)
    im = rescale(autofluor, (Z_scale, XY_scale, XY_scale), anti_aliasing=True, preserve_range=True)
    
    
    #%%
    ### OPTIONALLY IF DONT WANT TO LOAD DSET FROM N5 file

    if pad:
        reg_name = np.load(atlas_dir + 'brainreg_args.npy')
        z_after = int(reg_name[-1])
        x_after = int(reg_name[-2])
        
        # find actual down_factor while considering the padding on each side of the image
        m_shape = np.asarray(myelin.shape)
        m_shape[0] = m_shape[0] - z_after*2
        m_shape[1] = m_shape[1] - x_after*2
        m_shape[2] = m_shape[2] - x_after*2
    
        down_factor = dset.shape/m_shape
        
        # down_factor = [1526, 12422, 10674]/m_shape
        
    else:
        down_factor = dset.shape/np.asarray(myelin.shape)
        
        
    print('down_factor')
    print(down_factor)
    print(myelin.shape)
     
    
       
    coords_df['Z_down'] = (coords_df['Z_scaled'] - overlap_pz)/down_factor[0]
    coords_df['X_down'] = (coords_df['X_scaled'] - overlap_pxy)/down_factor[1]
    coords_df['Y_down'] = (coords_df['Y_scaled'] - overlap_pxy)/down_factor[2]
    
    
    if pad:
        # scale coords with padding if originally had padding for registration
        coords_df['Z_down'] = coords_df['Z_down'] + z_after
        coords_df['X_down'] = coords_df['X_down'] + x_after
        coords_df['Y_down'] = coords_df['Y_down'] + x_after
    





        
    #%% Loop through and get intensity of all larger cells as well to make a gated flow cytometry graph
    analysis_dir = exp['path'] + exp['name'] + '_MaskRCNN_patches/'
    images = glob.glob(os.path.join(analysis_dir,'*_df.pkl'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i.replace('_df.pkl','_input_im.tif'),pkl=i, segmentation=i.replace('_df.pkl','_segmentation_overlap3.tif'),
                     shifted=i.replace('_df.pkl', '_shifted.tif')) for i in images]
     
    
    


    #%%
    
    ### Read in atlas to get just the cortex
    atlas = tiff.imread(atlas_dir + '/registered_atlas.tiff')
    atlas = np.moveaxis(atlas, 0, 1)   ### reshuffle atlas so now in proper orientation
    atlas = np.flip(atlas, axis=0)  ### flip the Z-axis
    atlas = np.flip(atlas, axis=2)
    atlas_size_pre_resize = atlas.shape
    atlas = resize(atlas, myelin.shape, anti_aliasing=False, order=0, preserve_range=True)   ### rescale the images
    
    cc_allen = regionprops(atlas, cache=False)
    cc_labs_allen = [region['label'] for region in cc_allen]
    cc_labs_allen = np.asarray(cc_labs_allen)
    
    
    


    #%% Try getting ONLY the cortical cells first
    
    
    ### First start by defining everything we do NOT want
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
    remove_regions = np.zeros(np.shape(atlas))
    for idx in sub_ids:
        cur_id = np.where(cc_labs_allen == idx)[0]
        
        #print(cur_id)
        if len(cur_id) == 0:  ### if it does not exists in atlas
            continue
        cur_coords = cc_allen[cur_id[0]]['coords']
        remove_regions[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = idx
       

    iso_layer = np.copy(atlas)
    iso_layer[remove_regions > 0] = 0   ### delete all other regions
    # iso_layer[remove_regions == 0] = 0  ### keep current region


    tiff.imwrite(sav_dir + exp['name'] + '_removed_regions.tif', np.asarray(iso_layer, dtype=np.uint32))
    # import napari
    # viewer = napari.Viewer()
    
    
    # viewer.add_image(myelin)
    # viewer.add_image(iso_layer)
    # viewer.show(block=True)


    ### Then try splitting in white and grey matter
    coords_df['location'] = 'WM'
    cells_im, cell_pos = df_to_im(coords_df, atlas)
    plot_max(cells_im)
        
    loc_val = iso_layer[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]]
    
    ### show only GM cells
    coords_df.loc[np.where(loc_val > 0)[0], 'location'] = 'GM'
    
    large_GM = coords_df.iloc[np.where(coords_df['location'] == 'GM')[0]]
    
    cells_im_LARGE_GM, cell_pos = df_to_im(large_GM, atlas)
    plot_max(cells_im_LARGE_GM)
    
    

    if os.path.isfile(sav_dir + filename + '_CLEANED_LARGE_COORDS_df_CNN' + run_name + '.pkl') and CNN:
        
        print('SKIPPING - file already exists')
        
 
        # check_OLs_df = pd.read_pickle(sav_dir + filename + '_CLEANED_LARGE_COORDS_df.pkl')
        
        check_OLs_df = pd.read_pickle(sav_dir + filename + '_CLEANED_LARGE_COORDS_df_CNN' + run_name + '.pkl')
        # check_OLs_df = pd.read_pickle(sav_dir + filename + '_CLEANED_LARGE_COORDS_df_CNN_LOWTHRESH.pkl')
        

        
    elif os.path.isfile(sav_dir + filename + '_CLEANED_LARGE_COORDS_df_ViT' + run_name + '.pkl') and not CNN:
        
        print('SKIPPING - file already exists ViT')
        
        check_OLs_df = pd.read_pickle(sav_dir + filename + '_CLEANED_LARGE_COORDS_df_ViT' + run_name + '.pkl')
        
        
        

    else:
        
        #%% RUN UNET TO CLASSIFY CELLS
        # zzz
        
        crop_size = 40
        z_size = 8
        
    
        num_corr = 0
        # dataset_name = 'lower_300'
        # dataset_name = 'lower_500_stitchline'
        
        # for higher thresh
        # check_OLs_df = coords_df.iloc[np.where((coords_df['location'] == 'GM') & (coords_df['vols'] > 250))[0]]
        
        
        
        ### for lower thresh
        # check_OLs_df = coords_df.iloc[np.where((coords_df['location'] == 'GM') & (coords_df['vols'] > 150))[0]]
        
        
        ### better thresh
        check_OLs_df = coords_df.iloc[np.where((coords_df['location'] == 'GM') & (coords_df['vols'] > 200))[0]]
        
        
        # For striatum
        # check_OLs_df = coords_df.iloc[np.where((coords_df['location'] == 'GM') & (coords_df['vols'] > 200))[0]]
        # 
        print('Checking total: ' + str(len(check_OLs_df)))
    
        ### add additional thresh
        # check_OLs_df = coords_df.iloc[np.where((coords_df['location'] == 'GM') & (coords_df['vols'] > 400))[0]]

        check_OLs_df['prediction'] = -1
        check_OLs_df['confidence'] = -1
        
        cell_num = 1
        
        
        # import tqdm                                                                                                   
        # import concurrent.futures
        # import multiprocessing
        
        # def func_df(dset, x, y, z, crop_size, z_size):
        #     print(x)
        #     crop_in = dset[z-z_size:z+z_size, x-crop_size:x+crop_size, y-crop_size:y+crop_size]
            
            
        #     return crop_in
            
        # # Process the rows in chunks in parallel
        # with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        #     # df['result'] = list(tqdm.tqdm(pool.map(func, dset, check_OLs_df['X_dset'], check_OLs_df['Y_dset'], check_OLs_df['Z_dset'], 
        #     #                                        ), total=len(check_OLs_df))) # With a progressbar
                          
            
        #     num_cells = 10
        #     from multiprocessing import Pool, cpu_count
        #     pool = Pool(processes=4)
            
        #     result = pool.starmap(func_df, zip(dset, check_OLs_df.iloc[0:num_cells]['X_dset'], check_OLs_df.iloc[0:num_cells]['Y_dset'], 
        #                                            check_OLs_df.iloc[0:num_cells]['Z_dset'], check_OLs_df.iloc[0:num_cells]['crop_size'], check_OLs_df.iloc[0:num_cells]['z_size'],
        #                                            ))
            
            
        #     check_OLs_df['result'] = list(tqdm.tqdm(pool.map(func_df, dset, check_OLs_df.iloc[0:num_cells]['X_dset'], check_OLs_df.iloc[0:num_cells]['Y_dset'], 
        #                                            check_OLs_df.iloc[0:num_cells]['Z_dset'], check_OLs_df.iloc[0:num_cells]['crop_size'], check_OLs_df.iloc[0:num_cells]['z_size'],
        #                                            chunksize=10), total=len(check_OLs_df))) # With a progressbar
                                                    
                       
                        
                       
            
        check_OLs_df['X_dset'] = check_OLs_df['X_scaled'] - overlap_pxy
        check_OLs_df['Y_dset'] = check_OLs_df['Y_scaled'] - overlap_pxy
        check_OLs_df['Z_dset'] = check_OLs_df['Z_scaled'] - overlap_pz
        
        # check_OLs_df['crop_size']  = crop_size
        # check_OLs_df['z_size']  = z_size
        
        
        
        if not CNN:
            #%% Import vision transformer
            from predict import *
            
            
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CustomViTForImageClassification()
            #model_1,2,3
            #model.load_state_dict(torch.load(r"./model_1,2,3.pth",map_location=device))
            model.load_state_dict(torch.load(r"./model_1,2.pth",map_location=device))
            model = model.to(device)


            #if you using model 1,2,3 please use here
            #mean = 373.46493034947963
            #std = 168.13003627997972
            #if you using model 1,2please use here
            mean = 376.16833740270033
            std = 169.88323430016285


            ### OLD model and ViT file
            # from predict_LARGE_OL_ViT import *
            # model_path = 'best_val_loss_model.pth'
            # model = load_model(model_path, device)        
             
        
        
        #%% Run analysis
        import time
        tic = time.perf_counter()
        for id_r, row in check_OLs_df.iterrows():
            
            
            
            ### skip if already corrected before
            if check_OLs_df.loc[id_r,'prediction'] != -1:
                continue
            
            x = row['X_dset']
            y = row['Y_dset']
            z = row['Z_dset']
            
            crop_in = dset[z-z_size:z+z_size, x-crop_size:x+crop_size, y-crop_size:y+crop_size]
            
            crop = np.asarray(crop_in, dtype=np.float32)
            
            
            if cell_num % 1000 == 0:
                toc = time.perf_counter()
                print(f"Opened subblock in {toc - tic:0.4f} seconds")
                tic = time.perf_counter()
            
        
            #%% If want to use ViT instead
            if not CNN:
                # prediction, conf = predict(model, crop, device)
                # probs = conf
                # preds = prediction[0][0]
                prediction = predict(model, crop, mean, std, device)
                probs = 1
                # print("prediction is", prediction)
                if(prediction < 0.5):
                    # print("Prediction: ", 0)
                    preds = 0
                else:
                    # print("Prediction: ",1)
                    preds = 1

            #%% Else use UNet
            else:
            
                """ Normalization """
                crop = normalize(crop, mean_arr, std_arr)
                
                """ Analyze """
                """ set inputs and truth """
                crop = np.expand_dims(crop, axis=-1)
                batch_x = crop
                batch_x = np.moveaxis(batch_x, -1, 0)
                batch_x = np.expand_dims(batch_x, axis=0)
                
                """ Convert to Tensor """
                inputs_val = torch.tensor(batch_x, dtype = torch.float, device=device, requires_grad=False)
                #labels_val = torch.tensor(batch_y, dtype = torch.long, device=device, requires_grad=False)
        
                # forward pass to check validation
                output_val = unet(inputs_val)
                output_val = torch.squeeze(output_val)  ### have to squeeze to remove dimensions of size 1 after adaptiveavgpool3d()
                
                        
                ### get probabilities of output!
                probs = torch.nn.functional.softmax(output_val, dim=0)
                conf, indices = torch.max(probs, axis=0)
                
                """ Convert back to cpu """                                      
                output_val = output_val.cpu().data.numpy()            
                # output_val = np.moveaxis(output_val, 1, -1)
                preds = np.argmax(output_val)
                probs = conf.cpu().data.numpy()   ### get only probability for column 1
                
            
            #%% Save output
    
            ### Plot
            # plot_max(crop)
            # plt.title('Pred: ' + str(preds) + ' Prob: ' + str(np.round(probs, 2)) + ' Vol: ' + str(row['vols']), pad=None)
            # print(str(row['vols']))
            
            
            check_OLs_df.loc[id_r,'prediction'] = preds
            check_OLs_df.loc[id_r,'confidence'] = np.round(probs, 2)
            
            cell_num += 1
            if cell_num % 1000 == 0:     
                print('Checking cell num: ' + str(cell_num) + ' of total: ' + str(len(check_OLs_df)))
       
            
    
        ### SAVE the large_df
        if not CNN:
            check_OLs_df.to_pickle(sav_dir + filename + '_CLEANED_LARGE_COORDS_df_ViT' + run_name + '.pkl')
        else:
            check_OLs_df.to_pickle(sav_dir + filename + '_CLEANED_LARGE_COORDS_df_CNN' + run_name + '.pkl')
        # large_df = pd.read_pickle(sav_dir + filename + '_LARGE_COORDS_df.pkl')
    
        
    
    # import seaborn as sns
    # plt.figure()
    # sns.scatterplot(check_OLs_df, x='vols', y='confidence', hue='prediction', s=2)
    
    if not CNN:
        check_pred = check_OLs_df.iloc[np.where(check_OLs_df['prediction'] == 0)[0]]  ### first only get the gray matter cells

    else:
        check_pred = check_OLs_df.iloc[np.where(check_OLs_df['prediction'] == 1)[0]]  ### first only get the gray matter cells
    # sorted_df = sorted_df.sort_values(['vols', 'mean_int'], ascending=False)
    
    # plt.figure()
    # sns.scatterplot(data=sorted_df[0:1000], x="vols", y="mean_int", hue='is_OL', s=12, palette='Set2')
    # plt.pause(0.1)
               
    # sorted_df['is_OL'] = sorted_df['is_OL'].astype(int)
    
    cells_im_check, cell_pos = df_to_im(check_pred, atlas)
    plot_max(cells_im_check)
    
    
    if not CNN:
        tiff.imwrite(sav_dir + exp['name'] + '_LARGE_cells_ViT' + run_name + '.tif', np.asarray(cells_im_check, dtype=np.uint8))
        
    else:
        tiff.imwrite(sav_dir + exp['name'] + '_LARGE_cells_CNN' + run_name + '.tif', np.asarray(cells_im_check, dtype=np.uint8))


    cells_im_check[iso_layer == 0] = 0
    plot_max(cells_im_check)
    
    
    
    print(len(np.where(cells_im_check > 0)[0]))
    
    
    
    # ### Histogram:
    # check_OLs_df_small = coords_df.iloc[np.where((coords_df['location'] == 'GM') & (coords_df['vols'] <= 250))[0]]
    # vols_small = check_OLs_df_small['vols'].values
    
    # vols_small = np.concatenate((vols_small, check_OLs_df.iloc[np.where(check_OLs_df['prediction'] == 0)[0]]['vols'].values ))
    
    
    # vols_large = check_pred['vols'].values
    
    # plt.figure()
    # plt.hist(vols_small, bins=30)
    # plt.hist(vols_large, bins=30)
    
    
    #%% Find nearest neighbors and plot
    zzz
    
    
    ### SOME CELLS ARE MISSING FROM COORDS_DF??? Why are some nearest neighbor distances > 0???
    GM_cells = coords_df.iloc[np.where(coords_df['location'] == 'GM')[0]]
    
    all_points = GM_cells[['X_scaled', 'Y_scaled', 'Z_scaled']].values
    
    query_points = check_pred[['X_scaled', 'Y_scaled', 'Z_scaled']].values
    
    
    from scipy.spatial import KDTree
    # Build the KD tree
    kdtree = KDTree(all_points)
    
    # Query point
    # query_point = np.array([0.5, 0.5, 0.5])
    
    # Query the KD tree for the 5 nearest neighbors to the query point
    distances, indices = kdtree.query(query_points, k=5)
    
    all_comps = []
    for cell_num, inds in enumerate(indices):
        
        cur_neighbors = GM_cells.iloc[inds[1:]]['vols_um'].values
        # cur_point = GM_cells.iloc[inds[0]]['vols']   ### skip 0th index, which is just current point
        
        cur_point = check_pred.iloc[cell_num]['vols_um']

        all_comps.append({'Large':cur_point, 'Neighbor':np.mean(cur_neighbors)})
        
    df_neighbors = pd.DataFrame(all_comps)
    
    df_neighbors = pd.melt(df_neighbors)
    
    plt.figure(figsize=(3, 4)); 
    # sns.stripplot(df_neighbors, x='variable', y='value', jitter=False)
    
    # sns.boxplot(df_neighbors, x='variable', y='value')

    for row in all_comps:
        plt.plot(['Large', 'Neighbor'], [row['Large'], row['Neighbor']],
                 c='gray', linewidth=2, alpha=0.2)

 
    sns.pointplot(df_neighbors, x='variable', y='value', errorbar='sd', c='red', markersize=1, capsize=0.1,
                  marker='D', linewidth=1, linestyle='--')  
    
    ax = plt.gca()
    #sns.move_legend(ax, loc=leg_loc, frameon=False, title='', fontsize=fontsize)
    fontsize=14
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylabel('Volume (um^3)', fontsize=fontsize)
    plt.xlabel('', fontsize=fontsize)
    plt.tight_layout()

    sav_fold = '/media/user/8TB_HDD/Plot_outputs/'

    plt.savefig(sav_fold + exp['name'] +'_large_vs_neighbor.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp['name'] +'_large_vs_neighbor.svg', format='svg', dpi=300)
    
    ### Run stats    
    stats_large_OLs = pd.DataFrame(all_comps)
    
    res = stats.ttest_ind(stats_large_OLs['Large'], stats_large_OLs['Neighbor'], equal_var=True, alternative='two-sided')
    print(res.pvalue)
    print('Large:' + str(np.mean(stats_large_OLs['Large'])) + ' Neighbor: ' + str(np.mean(stats_large_OLs['Neighbor'])))
    print('Large:' + str(stats_large_OLs['Large'].sem()) + ' Neighbor: ' + str(stats_large_OLs['Neighbor'].sem()))
    
          
    ### Force calculate p value
    from mpmath import mp
    t = mp.mpf(res.statistic)
    nu = mp.mpf(res.df)
    x2 = nu / (t**2 + nu)
    p = mp.betainc(nu/2, mp.one/2, x2=x2, regularized=True)
    print(p)
    
    
    
    
    
    check = pd.DataFrame(all_comps)
    inverted = check.iloc[np.where(check['Large'] - check['Neighbor'] < 0)]
    print('number of cells where neighbors are larger: ' + str(len(inverted)))
    print('out of total: ' + str(len(check)))
    
    
    
    ### plot the inverted cells
    check_nearby = check_pred.iloc[np.where(check['Large'] - check['Neighbor'] < 0)]
    
    cells_im_nearby, cell_pos = df_to_im(check_nearby, atlas)
    plot_max(cells_im_nearby)
    
    
    
    ### DROP INVERTED NEIGHBOR CELLS
    check_pred = check_pred.iloc[np.where(check['Large'] - check['Neighbor'] > 0)]
    cells_im_dropped, cell_pos = df_to_im(check_pred, atlas)
    plot_max(cells_im_dropped)
    
    
    
    """
    
    
    
    
        DELETE CELLS IMMEDIATELY ADJACENT TO EACH OTHER!!! 
    
    
    
    
    
    """
    
    
     
    
    
    # zzz

    #%%
    """
    Syglass to MaMut Code Converter

    @author: Ephraim Musheyev for the Bergles Lab (2022)
    """
    print('Converting to MaMuT')
    import xml.etree.ElementTree as ET

    xy_scale = 1
    z_scale = 1

    """
    CONVERTER:
    """


   
    
    
    """ For testing ILASTIK images """
    xml_files = glob.glob(os.path.join(input_path,'*.xml'))    # can switch this to "*truth.tif" if there is no name for "input"
    xml_files.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    
    # xml_path = xml_files[0]
    
    ### path is shortest xml string
    xml_path = min(xml_files, key=len)
    xml_name = xml_path.split('/')[-1]

    
    # Reads in the syGlass CSV
    # df= pd.read_csv(syglass_path)
    
    shell_path = './MaMut_Shell_File_updated.xml' # <- Enter the "MaMut_Shell_File.xml" pathname here (as a string)


    xml_file_name =  exp['name']
    destination_path = sav_dir #+ exp['name'] 
    
    
    df = pd.DataFrame()
    
        
    
    df['Y'] = check_pred['X_dset']
    df['X'] = check_pred['Y_dset']
    df['Z'] = check_pred['Z_dset']
    
    df['SERIES'] = check_pred.index
    df['FRAME'] = 0
    df['COLOR'] = check_pred['prediction']   
    
    df['VOLUME'] = check_pred['vols']
    
    # df['Y'] = check_OLs_df['X_dset']
    # df['X'] = check_OLs_df['Y_dset']
    # df['Z'] = check_OLs_df['Z_dset']
    
    # df['SERIES'] = check_OLs_df.index
    # df['FRAME'] = 0
    # df['COLOR'] = check_OLs_df['prediction']   
    
    
    
    num_rows = df.shape[0] #gets passed to the AllSpots nspots element




    """
    Converts all the individual datapoints in the CSV to
    MaMut-style XML format

    Datapoints are arranged according to their timepoint
    (All datapoints at t=0 are grouped under one SpotsInFrame element)
    """
    def spots_conversion(df, xy_scale, z_scale):
        
        # Creating AllSpots element, which contains all SpotsInFrame elements
        AllSpots_element = ET.Element('AllSpots')
        AllSpots_element.set('nspots',str(num_rows))

        min_frame = df['FRAME'].min()
        max_frame = df['FRAME'].max()
        
        # Creates SpotsInFrame element for each timepoint
        for i in range(min_frame,max_frame+1):

            SpotsInFrame_element = ET.Element('SpotsInFrame')
            SpotsInFrame_element.set('frame',str(i))
            df_i = df[df['FRAME'] == i]

            # Creates Spots subelement, which contains all the datapoints at each SpotsInFrame timeframe
            for index, row in df_i.iterrows():
                
                # Extracts relevant data from CSV for each datapoint
                x=row['X'] * xy_scale
                y=row['Y'] * xy_scale
                z=row['Z'] * z_scale
                frame=row['FRAME']
                series=row['SERIES']  
                color=row['COLOR']

                """
                A Spot ID is assigned to each spot
                It is formed by adding a 0 the beginning of the frame number and series number, and combining the two numbers 
                (ex. frame 3 of series 34 would become "03034")
                """
                id_=(f'0{frame}0{series}')
                
                # Library for assigning colors to the spots later on
                blue='-13421569' # New cell
                red='-52429' # Old cell
                purple = '-3407668' #
                green='-13369549' # All else
                
                
                cell_volume = row['VOLUME']
                
                
                # Creating Spots subelement and setting its atributes
                Spots_element=ET.SubElement(SpotsInFrame_element, 'Spot')
            
                Spots_element.set('ID', f'{id_}')
                Spots_element.set('name',f'ID{id_}')  
                Spots_element.set('STD_INTENSITY_CH1','0.0')  
                Spots_element.set('STD_INTENSITY_CH2','0.0')  
                Spots_element.set('QUALITY','-1.0') 
                Spots_element.set('POSITION_T',f'{frame}') 
                Spots_element.set('TOTAL_INTENSITY_CH2','0.0') 
                Spots_element.set('TOTAL_INTENSITY_CH1','0.0') 
                Spots_element.set('CONTRAST_CH1','0.0') 
                Spots_element.set('FRAME',f'{frame}.0') 
                Spots_element.set('CONTRAST_CH2','0.0') 
                Spots_element.set('MEAN_INTENSITY_CH1',f'{cell_volume}') 
                Spots_element.set('MAX_INTENSITY_CH2','0.0') 
                Spots_element.set('MEAN_INTENSITY_CH2','0.0') 
                Spots_element.set('MAX_INTENSITY_CH1','0.0') 
                Spots_element.set('SOURCE_ID','0') 
                Spots_element.set('MIN_INTENSITY_CH2','0.0') 
                Spots_element.set('MIN_INTENSITY_CH1','0.0') 
                Spots_element.set('SNR_CH1','0.0') 
                Spots_element.set('SNR_CH2','0.0') 
                Spots_element.set('MEDIAN_INTENSITY_CH1','0.0') 
                Spots_element.set('VISIBILITY','1') 
                Spots_element.set('RADIUS','10.0')
                
                #Assigning color to the spots
                if color == 0:
                    Spots_element.set('MANUAL_SPOT_COLOR',f'{green}')
                elif color == 1:
                    Spots_element.set('MANUAL_SPOT_COLOR',f'{purple}')
                elif color == 2:
                    Spots_element.set('MANUAL_SPOT_COLOR',f'{purple}')
                else:
                    Spots_element.set('MANUAL_SPOT_COLOR',f'{green}')
        
                Spots_element.set('MEDIAN_INTENSITY_CH2','0.0') 
                Spots_element.set('POSITION_X',f'{x}') 
                Spots_element.set('POSITION_Y', f'{y}') 
                Spots_element.set('POSITION_Z',f'{z}') 
            
            AllSpots_element.append(SpotsInFrame_element)

        return AllSpots_element




    # Parse the MaMut shell xml file
    tree = ET.parse(shell_path) 
    root = tree.getroot()

    # Locate the Model element of the XML file; Generating and appending the AllSpots, AllTracks, and Filtered Tracks elements
    model_element = root.find('Model')
    AllSpots= spots_conversion(df, xy_scale, z_scale)
    model_element.append(AllSpots)
    # Editing the name of the MaMut file name within the shell file
    imagedata_element = root.find('.//ImageData')
    imagedata_element.set('filename', f'{xml_name}')

    # Saving the XML file in the destination folder
    tree.write(f'{input_path}/{xml_name}_CNNbool_{CNN}_annotations.xml')#, encoding='utf-8', xml_declaration=True)
    
    
    print(len(check_pred))
    

    # zzz

    #######################################################################################################################

           
    #%% MAKE DENSITY MAP:
    print('Creating density map')
            
    if not CNN:
        coords_large = check_OLs_df.iloc[np.where(check_OLs_df['prediction'] == 0)[0]]  ### first only get the gray matter cells

    else:
        coords_large = check_OLs_df.iloc[np.where(check_OLs_df['prediction'] == 1)[0]]  ### first only get the gray matter cells
    
    downsampled_points = np.asarray([coords_large['Z_down'], coords_large['X_down'], coords_large['Y_down']]).T
    # and then move columns (1 to 0)  --> all these steps just do the inverse of original steps
    downsampled_points[:, [1, 0, 2]] = downsampled_points[:, [0, 1, 2]]
    

    deformation_field_paths = [atlas_dir + 'deformation_field_0.tiff',
                                atlas_dir + 'deformation_field_1.tiff',
                                atlas_dir + 'deformation_field_2.tiff'
                              ]

    
    deformation_field = tiff.imread(deformation_field_paths[0])
    
    ### Flip Z axis
    # downsampled_points[:, 0] = deformation_field.shape[0] - downsampled_points[:, 0]
    downsampled_points[:, 1] = deformation_field.shape[1] - downsampled_points[:, 1]
    downsampled_points[:, 2] = deformation_field.shape[2] - downsampled_points[:, 2]
            

    atlas_resolution = [20, 20, 20]



    field_scales = [int(1000 / resolution) for resolution in atlas_resolution]
    points = [[], [], []]
    for axis, deformation_field_path in enumerate(deformation_field_paths):
        deformation_field = tiff.imread(deformation_field_path)
        print('hello')
        for point in downsampled_points:
            point = [int(round(p)) for p in point]
            points[axis].append(
                int(
                    round(
                        field_scales[axis] * deformation_field[point[0], point[1], point[2]]
                    )
                )
            )
            
    points = np.transpose(points)


    ### plot onto ref_atlas???
    points_ref = np.zeros(np.shape(ref_atlas))
    
    p = np.copy(points)
    
    
    ### num out of bounds:
    len(np.where(points[:, 0] >= np.shape(points_ref)[0])[0])    ### most of them in this axis --- brainstem area maybe??? or olfactory
    len(np.where(points[:, 1] >= np.shape(points_ref)[1])[0])
    len(np.where(points[:, 2] >= np.shape(points_ref)[2])[0])
    
    
    # remove all out of bounds
    p[np.where(p[:, 0] >= np.shape(points_ref)[0])[0], 0] = np.shape(points_ref)[0] - 1
    p[np.where(p[:, 1] >= np.shape(points_ref)[1])[0], 1] = np.shape(points_ref)[1] - 1
    p[np.where(p[:, 2] >= np.shape(points_ref)[2])[0], 2] = np.shape(points_ref)[2] - 1
      
    
    points_ref[p[:, 0], p[:, 1], p[:, 2]] = 1
    
    
    plt.figure(); plt.imshow(points_ref[300])
    plt.figure(); plt.imshow(ref_atlas[300])
    

    #%% Do density map as density per region instead
    vals = ref_atlas[p[:, 0], p[:, 1], p[:, 2]]
    
    vals = vals[np.where(vals != 0)[0]]  # drop zeros
    
    unique, counts = np.unique(vals, return_counts=True)
    
    density_atlas = np.zeros(np.shape(ref_atlas), dtype=np.float64)
    
    
    ### also add to keys_df
    keys_df['num_large_W_CLEAN'] = np.nan
    
    for p_id, region_id in enumerate(unique):
        
        if region_id not in sub_ids:  ### only analyze if in the sub_region atlas
        
            ### get size of region in microns so can scale
            coords = np.transpose(np.where(ref_atlas == region_id))
            cub_mm = len(coords) * 0.000008    ### currently 20 x 20 x 20 um per voxel ---> which is 0.02 * 0.02 * 0.02 mm per voxel
            
            num_cells = counts[p_id]
            
            density = num_cells/cub_mm
            
            density_atlas[coords[:, 0], coords[:, 1], coords[:, 2]] = density
            
            
            ### also add to keys_df
            keys_df.loc[np.where(keys_df['ids'] == region_id)[0], 'num_large_W_CLEAN'] = num_cells
            
            
        
        
    ### save density image
    tiff.imwrite(sav_dir + exp['name'] + '_LARGE_DENSITY_MAP.tif', np.asarray(density_atlas, dtype=np.uint32))

    
    # viewer = napari.Viewer()
    
    # viewer.add_image(density_atlas)
    # viewer.add_image(points_ref)
    # viewer.show(block=True)
    

    
    
    
    ### Just need to add "num_Large_W" and re-save!!!
    
    fold = exp['path'] + exp['name'] + '_postprocess'

    # coords = glob.glob(os.path.join(fold,'*_coords_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE.pkl'))    # can switch this to "*truth.tif" if there is no name for "input"
    keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE.pkl'))    

    keys_df_RAW = pd.read_pickle(keys[0])
    # coords_df_RAW = pd.read_pickle(coords[0])
    
    # print('number of cells: ' + str(len(coords_df)))


    ### Add in the new cleaned large OLs
    
    
    keys_df_RAW['num_large_W_CLEAN']  = keys_df['num_large_W_CLEAN'] 
    
    
    keys_df_RAW.to_pickle(keys[0])
    

   

















    """
#################################################################################################################################


    zzz



    
    # #%%%% IF WANT TO CONTINUE THE ANALYSIS LIKE IN postprocessCOMPARE (this is temporary for now) 
    
    # ### first sort by depth level and pool all values from children
    
    # keys_df = keys_df_RAW
    
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
          
    # keys_df = df_level
    
    # keys_df['density_W_LARGE_CLEAN'] = keys_df['num_large_W_CLEAN'] / keys_df['atlas_vol_W'] 
    # keys_df['density_W'] = keys_df['num_OLs_W']/keys_df['atlas_vol_W']

    # ### MUST DROP ZEROS!!! Areas initially excluded for large OL counting (WM, tracts, nuclei, hindbrain, ect...)
    
    # large_df = keys_df.iloc[np.where(keys_df['density_W_LARGE_CLEAN'] != 0)]



    # zzz
        
    
    
    # # sav_fold = '/media/user/8TB_HDD/Plot_outputs/'
    # fontsize = 14


    # plt.figure(figsize=(4, 4))
    # ax = plt.gca()
    # sns.regplot(x='density_W', y='density_W_LARGE_CLEAN', data=large_df.dropna(), scatter_kws={"color": "grey", 's':10}, line_kws={"color": "red", "alpha":0.2}, ax=ax)
    # r,p = stats.pearsonr(large_df.dropna()['density_W'], large_df.dropna()['density_W_LARGE_CLEAN'])
    # print(r)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    # plt.yticks(fontsize=fontsize - 2)
    # plt.xticks(fontsize=fontsize - 2)

    # # plt.xlim([0, 300000])
    # # plt.ylim([0, 1])
    # ax.ticklabel_format(axis='x', scilimits=[-3, 3])  ### set to be order of magnitude
    # ax.ticklabel_format(axis='y', scilimits=[-3, 3])
    
    # plt.xlabel('Density of OLs (cells/mm\u00b3)', fontsize=fontsize)
    # plt.ylabel('Density of large OLs (cells/mm\u00b3)', fontsize=fontsize)
    # plt.tight_layout()
    
    # # plt.savefig(sav_fold + 'COMPARE_MYELIN_intensity_density_vs_OL_density_Rval_' + str(np.round(r, 2)) + '.png', format='png', dpi=300)
    # # plt.savefig(sav_fold + 'COMPARE_MYELIN_intensity_density_vs_OL_density_Rval_' + str(np.round(r, 2)) + '.svg', format='svg', dpi=300)
    
    
        
        
        
        
        
        
        
        
    
    

        
    #%% #############################################################################################
        
    ### now get cells within specific range of data to correct
    
    
    # mask = np.zeros(np.shape(cells_im_check))
    
    mask = np.ones(np.shape(cells_im_check))
    
    # ### For M260 closer to WM
    # # x = 130:160
    # # y = 300:350
    # # z = 250:300
    # # mask[250:300, 300:350, 130:160] = 1
    
    
    ### M260 striatum?
    # x = 160:200
    # y = 350:400
    # z = 200:300
    mask[200:300, 350:400, 160:200] = 1
       
    ### M260 deep
    # x = 170:230
    # y = 280:350
    # z = 150:200
    mask[150:200, 280:350, 170:230] = 1
           
    
    ### M260 cortex
    # x = 160:200
    # y = 350:400
    # z = 350:400
    mask[350:400, 350:400, 160:200] = 1    
    
    
    
    # ### M271 - 22mos old brain - corpus callosum/RSP
    # # x = 345:370
    # # y = 450:490
    # # z = 300:400
    # # below is z, y, x
    # mask[300:400, 450:490, 345:370] = 1       

    
    # ### M271 - 22mos old brain - left hippocampal nearby?
    # # x = 142:165
    # # y = 477:500
    # # z = 300:400
    # # below is z, y, x
    # mask[300:400, 477:500, 142:165] = 1   


    # # ### M271 - 22mos old brain - right hippocampal white matter
    # # # x = 577:600
    # # # y = 512:550
    # # # z = 200:300
    # # # below is z, y, x
    # mask[200:300, 512:550, 577:600] = 1   
    
    # # ### M271 - 22mos old brain - prefrontal
    # # # x = 240:280
    # # # y = 212:244
    # # # z = 100:200
    # # # below is z, y, x
    # mask[100:200, 212:244, 240:280] = 1   
    
    # ### VALUE 5 -- WM NOT an OL
    # ### VALUE 6 -- WM YES an OL
            


    cells_match = mask[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]]
    match_ids = np.where(cells_match)[0]
    
    
    
    ### run below code if want to do all cells (NOT high conf)
    
    # match_ids = np.where(cells_match != 1)[0]  ### all OTHER cells NOT in mask

    print(len(match_ids))

    small_FOV_df = check_pred.iloc[match_ids]
    # small_FOV_df = check_pred_HC.iloc[match_ids]
    
    
    small_FOV_df['is_OL'] = -1
    # small_FOV_df = small_FOV_df.reset_index()


    import napari
    crop_size = 40
    z_size = 8
    

    num_corr = 0
    for id_r, row in small_FOV_df.iterrows():
        
        ### skip if already corrected before
        if small_FOV_df.loc[id_r,'is_OL'] != -1: #or small_FOV_df.loc[id_r, 'Z_down'] < 370:  
            continue
        
        # ### check error ones again
        # if sorted_df.loc[id_r,'is_OL'] != 0:
        #     num_corr += 1
        #     continue            
        
        viewer = napari.Viewer()
        
        x = row['X_scaled'] - overlap_pxy
        y = row['Y_scaled'] - overlap_pxy
        z = row['Z_scaled'] - overlap_pz
        
        crop = dset[z-z_size:z+z_size, x-crop_size:x+crop_size, y-crop_size:y+crop_size]
        
        # print(str(row['mean_int']) + ' and ' + str(row['vols']))
        
        print(str(row['vols']))
        print(str(row['confidence']))
        
        
        ### CHECK THESE LIMITS!!!
        viewer.add_image(crop, contrast_limits=(0, 1500))
        
        # viewer.add_image(crop, contrast_limits=(0, 3500))
        
        # viewer.add_image(crop, contrast_limits=(0, 1800))
        viewer.show(block=True)
        
        
        val = input("Press 1 if real OL, else 0")
        
        small_FOV_df.loc[id_r,'is_OL'] = int(val)
        
        if num_corr > 20:
            zzz
            
        num_corr += 1
        
    print('Total corrected: ' + str(len(np.where(small_FOV_df['is_OL']  != -1)[0])))

    ## SAVE the large_df
    # small_FOV_df.to_pickle(sav_dir + filename + '_SMALL_FOV_after_UNet_700_OLD_brain_near_WM.pkl')   ### _corrected_top_1000
    
    # small_FOV_df.to_pickle(sav_dir + filename + '_SMALL_FOV_after_UNet_248_OLD_brain_GM.pkl') 

    small_FOV_df.to_pickle(sav_dir + filename + '_SMALL_FOV_after_UNet_800_8mos_brain_GM.pkl') 






    #############################################################################################
    #%% Save crops
            
    
    crop_size = 40
    z_size = 8
    

    num_corr = 0
    # dataset_name = 'lower_300'
    # dataset_name = 'lower_500_stitchline'
    
    sorted_df = small_FOV_df
    
    
    # dataset_name = 'small_FOV_after_UNet_1000'
    
    # dataset_name = 'small_FOV_after_UNet_700_OLD_tissue_near_WM'
    
    # dataset_name = 'small_FOV_after_UNet_248_OLD_brain_GM'
    
    dataset_name = 'small_FOV_after_UNet_800_8mos_brain_GM'
    
    dataset_vals = sorted_df.iloc[np.where(sorted_df['is_OL'] != -1)[0]]
    
    for id_r, row in dataset_vals.iterrows():
        
        ### skip if already corrected before
        if sorted_df.loc[id_r,'is_OL'] == -1:
            continue
        
        x = row['X_scaled'] - overlap_pxy
        y = row['Y_scaled'] - overlap_pxy
        z = row['Z_scaled'] - overlap_pz
        
        crop = dset[z-z_size:z+z_size, x-crop_size:x+crop_size, y-crop_size:y+crop_size]
        
        ma = plot_max(crop, plot=0)
        
        # print(str(row['mean_int']) + ' and ' + str(row['vols']))
         
        print(str(row['vols']))
        

        tiff.imwrite(sav_dir + filename + '_' + dataset_name + '_CROP_LARGEOL_' + str(id_r) + '_val_' + str(sorted_df.loc[id_r,'is_OL']) + '.tif', crop)
        # tiff.imwrite(sav_dir + filename + '_' + dataset_name + '_MAX_PROJECT_CROP_LARGEOL_' + str(id_r) + '_val_' + str(sorted_df.loc[id_r,'is_OL']) + '.tif', ma)
            
        num_corr += 1
    
    
    dataset_vals.to_pickle(sav_dir + filename + '_LARGEOL_corrected_' + dataset_name + '.pkl')   ### _corrected_top_1000
    
    """
    
        
        
        
        
        
        # # zzz
        # # # load all pickle files and parse into larger df
        # all_pkl = []
        # large_only = []
        # #vols_only = []
        # for blk_num in range(len(examples)):
         
        #     #try:
        #     pkl = pd.read_pickle(examples[blk_num]['pkl'])
            
        #     if len(pkl) == 0:
        #         print('No cells in pickle?')
        #         continue
        #     vols = [len(coords) for coords in pkl['coords_scaled']]
            

        #     # if want just coords
        #     c_pkl = pkl[['X_scaled', 'Y_scaled', 'Z_scaled']].copy()
        #     c_pkl['vols'] = vols
            
            
        #     ### If volume is past threshold, then check intensity in raw data
            
        #     def get_im(dset, s_c, Lpatch_depth, Lpatch_size):
                
        #             #tic = time.perf_counter()
                    
        #             ### If nearing borders of image, prevent going out of bounds!
        #             z_top = s_c[2] + Lpatch_depth
        #             if z_top >= dset.shape[0]: z_top = dset.shape[0]
                    
        #             y_top = s_c[1] + Lpatch_size
        #             if y_top >= dset.shape[1]: y_top = dset.shape[1]
                    
        #             x_top = s_c[0] + Lpatch_size
        #             if x_top >= dset.shape[2]: x_top = dset.shape[2]
                    
        #             input_im = dset[s_c[2]:z_top, s_c[1]:y_top, s_c[0]:x_top]
        #             og_shape = input_im.shape
                    
        #             #toc = time.perf_counter()
        #             print('loaded asynchronously')
                    
        #             #print(f"Opened subblock in {toc - tic:0.4f} seconds")
                    
        #             return input_im, og_shape            
        

        #     s_c = pkl['xyz_offset'][0]

        #     Lpatch_size = 128 * 10
        #     Lpatch_depth = 64 * 4
        #     input_im, og_shape = get_im(dset, s_c, Lpatch_depth, Lpatch_size)
            
            
        #     large = np.where(c_pkl['vols'] > 250)[0]
            
        #     # large_im = np.zeros(np.shape(input_im))
                    
            
        #     # mean_int = []
        #     # for L_id in large:
                
        #     #     c = pkl.iloc[L_id]['coords_raw']
        #     #     c = np.copy(c)
                
        #     #     # large_coords[:, 0] = large_coords[:, 0] - pkl['overlap_pxy'][0]
        #     #     # large_coords[:, 1] = large_coords[:, 1] - pkl['overlap_pxy'][0]
        #     #     # large_coords[:, 2] = large_coords[:, 2] - pkl['overlap_pz'][0]
                
                
        #     #     c[:, 0] = c[:, 0] - pkl['overlap_pz'][0]
        #     #     c[:, 0][c[:, 0]  >= np.shape(input_im)[0]] = np.shape(input_im)[0] - 1
                
                
        #     #     c[:, 1] = c[:, 1] - pkl['overlap_pxy'][0]
        #     #     c[:, 1][c[:, 1]  >= np.shape(input_im)[1]] = np.shape(input_im)[1] - 1
                
        #     #     c[:, 2] = c[:, 2] - pkl['overlap_pxy'][0]
        #     #     c[:, 2][c[:, 2]  >= np.shape(input_im)[2]] = np.shape(input_im)[2] - 1
                        
        #     #     # large_im[c[:, 0], c[:, 1], c[:, 2]] = L_id
                
        #     #     mean = np.max(input_im[c[:, 0], c[:, 1], c[:, 2]])
                
        #     #     mean_int.append(mean)
                
        
        #     # large_pkl = c_pkl.iloc[large]
        #     # large_pkl['mean_int'] = mean_int
            
        #     large_only.append(large_pkl)
           
        #     print(blk_num)
        
        
        # large_df = pd.concat(large_only)


        ### need to reset the index here???

        
        ### scale the volumes
        
        
        
        # large_df['vols_um'] = large_df['vols'] * 1/res_diff   ### scale to isotropic
        # large_df['vols_um'] = large_df['vols_um'] / XY_res
        
        # large_df['Z_down'] = (large_df['Z_scaled'] - overlap_pz)/down_factor[0]
        # large_df['X_down'] = (large_df['X_scaled'] - overlap_pxy)/down_factor[1]
        # large_df['Y_down'] = (large_df['Y_scaled'] - overlap_pxy)/down_factor[2]
        
        
        # if pad:
        #     # scale coords with padding if originally had padding for registration
        #     large_df['Z_down'] = large_df['Z_down'] + z_after
        #     large_df['X_down'] = large_df['X_down'] + x_after
        #     large_df['Y_down'] = large_df['Y_down'] + x_after
        
        # # plt.figure()
        # # sns.scatterplot(data=large_df, x="vols", y="mean_int", size=1)
        
        
        
        # ### SAVE the large_df
        # large_df.to_pickle(sav_dir + filename + '_LARGE_COORDS_df.pkl')








        # #%%% find stitch lines
        # num_tiles = 6
        # pad = 50
        # stitch_spacing = np.round((atlas.shape[-1] - pad * 2)/num_tiles)
        # fov_width = np.round(1920/down_factor[-1])
        # width_stitch = int(fov_width * 0.1)  ### should be 10% of each FOV
        
        # stitch_im = np.zeros(np.shape(atlas))
        # stitch_adjacent = np.zeros(np.shape(atlas))
        # for line in range(1, num_tiles):
            
        #     center = np.asarray((stitch_spacing * line), dtype=int) + pad
            
        #     stitch_im[:, :, center-width_stitch : center + width_stitch] = 1
            
            
        #     stitch_adjacent[:, :, center - width_stitch * 3 : center - width_stitch * 2] = 1
        #     stitch_adjacent[:, :, center + width_stitch * 2 : center + width_stitch * 3] = 1
        
        
        # # viewer = napari.Viewer()
        
        # # viewer.add_image(np.max(cells_im, axis=0))
        # # viewer.add_image(np.max(stitch_im, axis=0))
        # # viewer.add_image(np.max(stitch_adjacent, axis=0))
        # # viewer.show(block=True)
                
        # ### compare mean vol in stitched areas relative to stitch adjacent to get fold factor for scaling
        # cell_pos = [coords_df['Z_down'].astype(int), coords_df['X_down'].astype(int), coords_df['Y_down'].astype(int)]
        # cell_pos = np.transpose(np.asarray(cell_pos))
        
        # ### make sure coordinates stay in range
        # cell_pos[np.where(cell_pos[:, 0] >= atlas.shape[0])[0], 0] = atlas.shape[0] - 1
        # cell_pos[np.where(cell_pos[:, 1] >= atlas.shape[1])[0], 1] = atlas.shape[1] - 1
        # cell_pos[np.where(cell_pos[:, 2] >= atlas.shape[2])[0], 2] = atlas.shape[2] - 1 
        
        # cells_in_stitch = np.where(stitch_im[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]])[0]
        # cells_adjacent = np.where(stitch_adjacent[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]])[0]
   
        
        # mean_stitch = np.mean(coords_df.iloc[cells_in_stitch]['vols'])
        # mean_adj = np.mean(coords_df.iloc[cells_adjacent]['vols'])
        
        
        # scale_factor = mean_adj/mean_stitch
        
        
        
        # #%%% Now compare before/after scale
        # cell_pos = [large_df['Z_down'].astype(int), large_df['X_down'].astype(int), large_df['Y_down'].astype(int)]
        # cell_pos = np.transpose(np.asarray(cell_pos))
        
        # ### make sure coordinates stay in range
        # cell_pos[np.where(cell_pos[:, 0] >= atlas.shape[0])[0], 0] = atlas.shape[0] - 1
        # cell_pos[np.where(cell_pos[:, 1] >= atlas.shape[1])[0], 1] = atlas.shape[1] - 1
        # cell_pos[np.where(cell_pos[:, 2] >= atlas.shape[2])[0], 2] = atlas.shape[2] - 1 
        
        # cells_in_stitch = np.where(stitch_im[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]])[0]
        # cells_adjacent = np.where(stitch_adjacent[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]])[0]
   
        
   
    
        # high_thresh = 500
    
        # scale_df = large_df.copy()
        
        # scale_df.loc[cells_in_stitch, 'vols'] = scale_df.loc[cells_in_stitch, 'vols'] * scale_factor
        
        # ### plot again *** WITH SCALING
        # scale_df['is_OL'] = -1
        # sorted_df = scale_df.iloc[np.where((scale_df['location'] == 'GM') & (scale_df['vols'] > high_thresh))[0]]  ### first only get the gray matter cells
        # sorted_df = sorted_df.sort_values(['vols', 'mean_int'], ascending=False)
        
        # plt.figure()
        # sns.scatterplot(data=sorted_df[0:1000], x="vols", y="mean_int", hue='is_OL', s=12, palette='Set2')
        # plt.pause(0.1)
                   
        # sorted_df['is_OL'] = sorted_df['is_OL'].astype(int)
        
        # cell_pos = [sorted_df['Z_down'].astype(int), sorted_df['X_down'].astype(int), sorted_df['Y_down'].astype(int)]
        # cell_pos = np.transpose(np.asarray(cell_pos))
        
        # ### make sure coordinates stay in range
        # cell_pos[np.where(cell_pos[:, 0] >= atlas.shape[0])[0], 0] = atlas.shape[0] - 1
        # cell_pos[np.where(cell_pos[:, 1] >= atlas.shape[1])[0], 1] = atlas.shape[1] - 1
        # cell_pos[np.where(cell_pos[:, 2] >= atlas.shape[2])[0], 2] = atlas.shape[2] - 1 
        
        # print('saving cells image')
        # cells_im_scaled = np.zeros(np.shape(atlas))
        # cells_im_scaled[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]] = 255  


        # # import napari
        # # viewer = napari.Viewer()
        # # viewer.add_image(cells_im)
        # # viewer.add_image(stitch_im)
        # # viewer.show(block=True)
        
        
        # ### plot again  *** WITHOUT SCALING
        # large_df['is_OL'] = -1
        # sorted_df = large_df.iloc[np.where((large_df['location'] == 'GM') & (large_df['vols'] > high_thresh))[0]]  ### first only get the gray matter cells
        # sorted_df = sorted_df.sort_values(['vols', 'mean_int'], ascending=False)
        
        # plt.figure()
        # sns.scatterplot(data=sorted_df[0:1000], x="vols", y="mean_int", hue='is_OL', s=12, palette='Set2')
        # plt.pause(0.1)
                   
        # sorted_df['is_OL'] = sorted_df['is_OL'].astype(int)
        
        # cell_pos = [sorted_df['Z_down'].astype(int), sorted_df['X_down'].astype(int), sorted_df['Y_down'].astype(int)]
        # cell_pos = np.transpose(np.asarray(cell_pos))
        
        # ### make sure coordinates stay in range
        # cell_pos[np.where(cell_pos[:, 0] >= atlas.shape[0])[0], 0] = atlas.shape[0] - 1
        # cell_pos[np.where(cell_pos[:, 1] >= atlas.shape[1])[0], 1] = atlas.shape[1] - 1
        # cell_pos[np.where(cell_pos[:, 2] >= atlas.shape[2])[0], 2] = atlas.shape[2] - 1 
        
        # print('saving cells image')
        # cells_im = np.zeros(np.shape(atlas))
        # cells_im[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]] = 255        
        
        # import napari
        # viewer = napari.Viewer()
        # new_layer = viewer.add_image(np.max(cells_im_scaled, axis=0), name='scaled')
        # new_layer = viewer.add_image(np.max(cells_im, axis=0), name='original')
        # new_layer = viewer.add_image(np.max(stitch_im, axis=0), name='stitch')
        # viewer.add_image(np.max(myelin, axis=0), name='stitch')
        # viewer.add_image(cells_im)
        # viewer.add_image(cells_im_scaled)
        # viewer.add_image(myelin)
        
        
        # new_layer = viewer.add_image(np.max(cells_im_check, axis=0), name='stitch')
        # viewer.add_image(cells_im_check)
        
        # viewer.show(block=True)
        
        
        
        
                
        # #%% Loop through and see what these cells are like and which ones are real and which ones are not
        # zzz
        # ## if want to load corrected pickle!!!
        # sorted_df = pd.read_pickle(sav_dir + filename + '_LARGE_COORDS_df_corrected_GM_lower_300.pkl')
        
        
        # # ### loop by first largest and brightest first
        
        # # ### also add columns to the dataframe to indicate if it has been analyzed or not
        
        # sorted_df = coords_df  ### if want to just send in the entire dataframe and parse by size later!
        # sorted_df['is_OL'] = -1
        # sorted_df['in_stitch'] = -1
        
        
        # #%%% Now compare before/after scale
        # cell_pos = [sorted_df['Z_down'].astype(int), sorted_df['X_down'].astype(int), sorted_df['Y_down'].astype(int)]
        # cell_pos = np.transpose(np.asarray(cell_pos))
        
        # ### make sure coordinates stay in range
        # cell_pos[np.where(cell_pos[:, 0] >= atlas.shape[0])[0], 0] = atlas.shape[0] - 1
        # cell_pos[np.where(cell_pos[:, 1] >= atlas.shape[1])[0], 1] = atlas.shape[1] - 1
        # cell_pos[np.where(cell_pos[:, 2] >= atlas.shape[2])[0], 2] = atlas.shape[2] - 1 
        
        # cells_in_stitch = np.where(stitch_im[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]])[0]
        
        
        # sorted_df.loc[sorted_df.iloc[cells_in_stitch].index, 'in_stitch'] = 1
        

        # sorted_df = sorted_df.iloc[np.where(sorted_df['location'] == 'GM')[0]]  ### first only get the gray matter cells
                
        # # sorted_df = sorted_df.sort_values(['vols', 'mean_int'], ascending=False)
        
        # import napari
        # crop_size = 40
        # z_size = 8
        

        # num_corr = 0
        # for id_r, row in sorted_df.iterrows():
            
        #     ### skip if already corrected before
        #     if sorted_df.loc[id_r,'is_OL'] != -1:  
        #         continue
            
        #     # if sorted_df.loc[id_r,'vols'] < 500: ### do some big ones for training
        #     #     continue
            
        #     if sorted_df.loc[id_r,'vols'] > 400 or sorted_df.loc[id_r,'vols'] < 300: ### do some small ones as well for training
        #         continue
            
            
        #     if sorted_df.loc[id_r, 'in_stitch'] == -1:   ### -1 means NOT in stitch
        #         continue
            
        #     ### check error ones again
        #     # if sorted_df.loc[id_r,'is_OL'] != 0:
        #     #     num_corr += 1
        #     #     continue            
            
        #     viewer = napari.Viewer()
            
        #     x = row['X_scaled'] - overlap_pxy
        #     y = row['Y_scaled'] - overlap_pxy
        #     z = row['Z_scaled'] - overlap_pz
            
        #     crop = dset[z-z_size:z+z_size, x-crop_size:x+crop_size, y-crop_size:y+crop_size]
            
        #     # print(str(row['mean_int']) + ' and ' + str(row['vols']))
            
        #     print(str(row['vols']))
            
            
        #     viewer.add_image(crop, contrast_limits=(0, 1500))
        #     viewer.show(block=True)
            
            
        #     val = input("Press 1 if real OL, else 0")
            
        #     sorted_df.loc[id_r,'is_OL'] = int(val)
            
        #     if num_corr > 20:
        #         zzz
                
        #     num_corr += 1
            
        # print('Total corrected: ' + str(len(np.where(sorted_df['is_OL']  != -1)[0])))

        # ## SAVE the large_df
        # sorted_df.to_pickle(sav_dir + filename + '_LARGE_COORDS_df_corrected_GM_lower_500_stitchline.pkl')   ### _corrected_top_1000
        
        
        # plt.figure()
        # sns.scatterplot(data=sorted_df[0:1000], x="vols", y="mean_int", hue='is_OL', s=12, palette='bright')
        # plt.pause(0.1)
                   
        # sorted_df['is_OL'] = sorted_df['is_OL'].astype(int)
        
        
        
        
        
        
        
        # # cell_pos = [sorted_df[0:1000]['Z_down'].astype(int), sorted_df[0:1000]['X_down'].astype(int), sorted_df[0:1000]['Y_down'].astype(int)]
        # # cell_pos = np.transpose(np.asarray(cell_pos))
        
        # # ### make sure coordinates stay in range
        # # cell_pos[np.where(cell_pos[:, 0] >= atlas.shape[0])[0], 0] = atlas.shape[0] - 1
        # # cell_pos[np.where(cell_pos[:, 1] >= atlas.shape[1])[0], 1] = atlas.shape[1] - 1
        # # cell_pos[np.where(cell_pos[:, 2] >= atlas.shape[2])[0], 2] = atlas.shape[2] - 1 
        
        # # print('saving cells image')
        # # cells_im = np.zeros(np.shape(atlas))
        # # cells_im[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]] = 255        
        
        
        # # viewer = napari.Viewer()
        
        # # viewer.add_image(cells_im)
        # # viewer.show(block=True)
        
        
        
        

### DEBUG: 
# import napari
# viewer = napari.Viewer()
# viewer.add_image(sphere_im)
    





