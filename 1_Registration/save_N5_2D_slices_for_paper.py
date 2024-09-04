# -*- coding: utf-8 -*-

import glob, os

# os_windows = 0
# if os.name == 'nt':  ## in Windows
#      os_windows = 1;
#      print('Detected Microsoft Windows OS')
# else: print('Detected non-Windows OS')


import numpy as np
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import tifffile as tiff
import z5py

"""  Network Begins: """

#input_path = '/media/user/Tx_LS_Data_13/20230704_M215_MoE_5xFAD_RI_1497_RIMS_5x_40perc_laser/BigStitcher/fused/'

# input_path = '/media/user/c0781205-1cf9-4ece-b3d5-96dd0fbf4a78/20231012_M223_MoE_Ai9_SHIELD_CUBIC_RIMS_RI_1500_3days_5x/M223_fused/'

# input_path = '/media/user/c0781205-1cf9-4ece-b3d5-96dd0fbf4a78/20231012_M230_MoE_PVCre_SHIELD_delip_RIMS_RI_1500_3days_5x/M230_fused/'

# input_path = '/media/user/c0781205-1cf9-4ece-b3d5-96dd0fbf4a78/20231013_M228_MoE_PVCre_SHIELD_delip_EasyIndex_RI_1520_3days_5x/M228_EasyIndex_FUSED/'


### 20x data
#input_path = '/media/user/4TB_SSD/20240210_M254_MoE_P60_low_AAVs_20x_RIMS_RI_1493_sunflow/Medium_FOV/'     
                  
                   
### 5x imaging of CUPRIZONE brain for comparison                  
#input_path = '/media/user/20TB_HDD_NAS1/20240215_M266_MoE_CUPRIZONE_6wks_SHIELD_CUBIC_7d_RIMS_RI_1493_sunflow/'

### slice 1031 in CUPRIZONE, and slice 1136 in M254 MoE control below (not age matched but looks nice)


# 5x imaging of sparse AAV labels
# input_path = '/media/user/20TB_HDD_NAS1/20240210_M254_MoE_P60_low_AAVs_SHIELD_CUBIC_7d_RIMS_2d_RI_1493_sunflower/'


# For RECOVERY - slice 1030 +/- 10 for max projection
# input_path = '/media/user/20TB_HDD_NAS_2/20240419_M312_REDO_REDO_PB_washed_GOOD_SHIELD_CUBIC_7d_RI_RIMS_14968_100perc_488_50perc_638_100msec/'



import sys
sys.path.append("..")
from get_brain_metadata import *


# list_brains = get_metadate(mouse_num = ['M115', 'M299', 'M138', 'M279', 'M271'])  ### for aging
# list_brains = get_metadata(mouse_num = ['M254', 'M265', 'M312'])    ### P60, cup, and Recovery
# list_brains = get_metadata(mouse_num = ['M147', 'M170'])    ### FVB, CD1



### updated

# list_brains = get_metadate(mouse_num = ['M260', 'M286', 'M271', 'Otx6'])  ### for aging
# list_brains = get_metadata(mouse_num = ['M260', 'M265', 'M312'])    ### P60, cup, and Recovery   

### Slice was about 250 WITH padding on low res so 250 - 50 pad == 200 * downsample factor 4 == 800

# list_brains = get_metadata(mouse_num = ['M147', 'M170'])    ### FVB, CD1


# list_brains = get_metadata(mouse_num = ['Otx6'])    ### FVB, CD1



# list_brains = get_metadata(mouse_num = ['M260'])    ### SLM and Hilus

list_brains = get_metadata(mouse_num = ['M271'])    ### SLM and Hilus




XY_res = 1.152035240378141
Z_res = 5


cloudreg = 0
ANTS = 1

# %% Parse the json file so we can choose what we want to extract or mask out
# with open('../atlas_ids/atlas_ids.json') as json_file:
#     data = json.load(json_file)
# data = data['msg'][0]


# reference_atlas = '/home/user/.brainglobe/633_perens_lsfm_mouse_20um_v1.0/annotation.tiff'

reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation.tiff'

# reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation_10.nrrd'

ref_atlas = tiff.imread(reference_atlas)
# ref_atlas = np.asarray(ref_atlas, dtype=np.uint32)
# right_ref_atlas = np.copy(ref_atlas)
# right_ref_atlas[:, :, 0:int(ref_atlas.shape[-1]/2)] = 0



ref_boundaries = '/home/user/.brainglobe/allen_mouse_20um_v1.2/boundaries.tif'
ref_bounds = tiff.imread(ref_boundaries)
ref_bounds = np.asarray(ref_bounds, dtype=np.uint32)


#%% Also extract isotropic volumes from reference atlas DIRECTLY - to avoid weird expansion factors 
# print('Extracting main key volumes')
# keys_dict = get_ids_all(data, all_keys=[], keywords=[''])  
# main_keys = pd.DataFrame.from_dict(keys_dict)
# main_keys = get_atlas_isotropic_vols(main_keys, ref_atlas, atlas_side='_W', XY_res=20, Z_res=20)
# main_keys['atlas_vol_R'] = main_keys['atlas_vol_W']/2
# main_keys['atlas_vol_L'] = main_keys['atlas_vol_W']/2

# keys_df = main_keys.copy(deep=True)


# sav_fold = '/media/user/8TB_HDD/Plot_SLICES/'

# density_fold = '/media/user/8TB_HDD/Mean_autofluor/'
sav_fold = '/media/user/8TB_HDD/Plot_HIGHRES_SLICES/'


for fold in list_brains:
    input_path = fold['path']
    name_str = fold['name']
    
    exp_name = fold['exp']
    # if exp_name == 'Cuprizone':
    #     exp_name = 'CUPRIZONE'
    
    # if exp_name == 'Recovery':
    #     exp_name = 'RECOVERY'
   
    
    n5_file = input_path + name_str + '.n5'
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
    #atlas_dir = downsampled_dir + name_str + '633_princeton_mouse_20u
    
    # actual everything, no gauss, -15 GRID
    ### NEW WITH ALLEN
    if fold['thresh'] != 0: ### for DIVIDED BY MYELIN
    
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




    allen_dir = downsampled_dir + name_str + '_ISOCORTEX_CORTEX_ONLY_allen_mouse_10um_bend_0.95_grid_-10_gauss_0/'
    analysis_dir = input_path + name_str + '_MaskRCNN_patches/'
    
    
    myelin_path = glob.glob(os.path.join(downsampled_dir,'*_ch0_n4_down1_resolution_20_PAD.tif'))[0]    # can switch this to "*truth.tif" if there is no name for "input"
      
    
    n5_file = input_path + name_str + '.n5'
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
     
     
    # input_name = examples[i]['input']  
    # filename = input_name.split('/')[-1].split('.')[0:-1]
    # filename = '.'.join(filename)        
       


    #%% To extract LARGE CROPS
    ## Extract channel 0    
    # with z5py.File(input_name, "r") as f:
    #     dset = f['setup0/timepoint0/' + 's0']
    #     crop_slice = dset[1013]
    #     dset = np.asarray(crop_slice, np.uint16)

    # tiff.imsave(sav_dir + filename + '_' + str(int(i)) + '_slice_1013_2D.tif', dset)
    
    
    ### DO IT BY DEPTH NOW
    # with z5py.File(input_name, "r") as f:
    #     dset = f['setup0/timepoint0/' + 's0']
 
    #     slices = [300, 600, 900, 1200, 1500]
    #     crop_size = 1000
    #     for i_slice in slices:
    #         crop_slice = dset[i_slice, 5500:5500 + crop_size, 6100:6100 + crop_size]
    #         crop = np.asarray(crop_slice, np.uint16)

    #         tiff.imsave(sav_dir + filename + '_DEPTH_progression_' + str(int(i_slice))  + '_2D.tif', crop)
    
    
    
    #%% For cuprizone
    ## Extract cuprizone brain   
    # with z5py.File(input_name, "r") as f:
    #     dset = f['setup0/timepoint0/' + 's0']
    #     crop_slice = dset[1031-10:1031+10]    # get small volume crop for max projection
    #     dset = np.asarray(crop_slice, np.uint16)

    # tiff.imsave(sav_dir + filename + '_' + str(int(i)) + '_slice_1031_FOR_CUPRIZONE_COMPARISON_3D.tif', dset)
            

    ## Extract CONTROL brain   
    # with z5py.File(input_name, "r") as f:
    #     dset = f['setup0/timepoint0/' + 's0']
    #     crop_slice = dset[1136-10:1136+10]    # get small volume crop for max projection
    #     dset = np.asarray(crop_slice, np.uint16)

    # tiff.imsave(sav_dir + filename + '_' + str(int(i)) + '_slice_1136_FOR_CUPRIZONE_COMPARISON_3D.tif', dset)
            
 
    
    ## Extract RECOVERY brain   
    # slice_v = 1030
    # slice_v = 800
    # with z5py.File(n5_file, "r") as f:
    #     dset = f['setup0/timepoint0/' + 's0']
    #     crop_slice = dset[slice_v-10:slice_v+10]    # get small volume crop for max projection
    #     crop_save = np.asarray(crop_slice, np.uint16)

    # tiff.imsave(sav_fold + name_str + '_slice_' + str(slice_v) + '_FOR_RECOVERY_COMPARISON_3D.tif', crop_save)
            

    # zzz
    
        
    ##%% Extract OLD BRAIN BLOBS   
    # slice_v = 800
    # with z5py.File(n5_file, "r") as f:
    #     dset = f['setup0/timepoint0/' + 's0']
    #     crop_slice = dset[slice_v-1:slice_v+1]    # get small volume crop for max projection
    #     crop_save = np.asarray(crop_slice, np.uint16)

    # tiff.imsave(sav_fold + name_str + '_slice_' + str(slice_v) + '_FOR_OLD_BRAIN_BLOBS.tif', crop_save)
            
    # slice_v = 800
    # with z5py.File(n5_file, "r") as f:
    #     dset = f['setup1/timepoint0/' + 's0']
    #     crop_slice = dset[slice_v-1:slice_v+1]    # get small volume crop for max projection
    #     crop_save = np.asarray(crop_slice, np.uint16)

    # tiff.imsave(sav_fold + name_str + '_slice_' + str(slice_v) + '_FOR_OLD_BRAIN_BLOBS_CHANNEL2.tif', crop_save)
            
    
    
    
    
    #%% Extract SLM and Hilus
    # slice_SLM = 1200; slice_hilus = 1092   ### for M260 (P60)
    slice_SLM = 1250; slice_hilus = 1140   ### for M271 (P620)
    with z5py.File(n5_file, "r") as f:
        dset = f['setup0/timepoint0/' + 's0']
        crop_slice = dset[slice_SLM]    # get small volume crop for max projection
        crop_save = np.asarray(crop_slice, np.uint16)
        tiff.imsave(sav_fold + name_str + '_slice_' + str(slice_SLM) + '_SLM_HIPPO.tif', crop_save)
                
        
        
        crop_slice = dset[slice_hilus]    # get small volume crop for max projection
        crop_save = np.asarray(crop_slice, np.uint16)
        tiff.imsave(sav_fold + name_str + '_slice_' + str(slice_hilus) + '_Hilus_HIPPO.tif', crop_save)
             

            
    


    zzz
        
        
    #%% Also get corresponding atlas
        
    
    #%% Get registered atlas and shift the axes
    from skimage.transform import rescale, resize, downscale_local_mean

    myelin = tiff.imread(myelin_path)
    
    atlas = tiff.imread(atlas_dir + '/registered_atlas.tiff')
    atlas = np.moveaxis(atlas, 0, 1)   ### reshuffle atlas so now in proper orientation
    atlas = np.flip(atlas, axis=0)  ### flip the Z-axis
    atlas = np.flip(atlas, axis=2)
    atlas_size_pre_resize = atlas.shape
    atlas = resize(atlas, myelin.shape, anti_aliasing=False, order=0, preserve_range=True)   ### rescale the images
    
    
    
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
    
    
    
    atlas_nopad = atlas[z_after:-z_after, x_after:-x_after, x_after:-x_after]
    slice_low = slice_v/down_factor[0]
    
    ref_low = atlas_nopad[int(slice_low)]
    
    tiff.imsave(sav_fold + name_str + '_ATLAS_low_res.tif', ref_low)
       
    
    #%% Create boundary image for reference atlas
    from skimage.segmentation import find_boundaries
    
    # bound_slice = np.zeros(np.shape(ref_low))
    # for slice_num, slice_im in enumerate(ref_low):
    bw_slice = find_boundaries(ref_low)
        # bound_slice[slice_num, :, :] = bw_slice

    bound_slice = np.asarray(bw_slice, dtype=np.uint8)
    tiff.imsave(sav_fold + name_str + '_boundaries_low_res.tif', bw_slice)
       
    
    ### REMOVE PADDING
    #atlas_nopad = atlas[z_after:-z_after, x_after:-x_after, x_after:-x_after]
    # right_nopad = right_atlas[z_after:-z_after, x_after:-x_after, x_after:-x_after]
    # left_nopad = left_atlas[z_after:-z_after, x_after:-x_after, x_after:-x_after]
    
    # zzz
        
    
    
    # #%% get small volume from 20x image to do max projection of myelinated segments!    
    # with z5py.File(input_name, "r") as f:
    #     dset = f['setup0/timepoint0/' + 's0']
 
    #     crop_vol = dset[650:690, 4200:5200, 4500:6900]
        
        
    #     crop = np.asarray(crop_vol, np.uint16)

    #     tiff.imsave(sav_dir + filename + '_VOL_20x_for_maxproj_C1.tif', crop)
        
    #     dset = f['setup1/timepoint0/' + 's0']
 
    #     crop_vol = dset[650:690, 4200:5200, 4500:6900]
    #     crop = np.asarray(crop_vol, np.uint16)

    #     tiff.imsave(sav_dir + filename + '_VOL_20x_for_maxproj_C2.tif', crop)    
    

        
            
    
# print('\n\nSegmented outputs saved in folder: ' + sav_dir)





