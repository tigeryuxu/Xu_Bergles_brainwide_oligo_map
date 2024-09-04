# -*- coding: utf-8 -*-

from skimage.transform import rescale, resize, downscale_local_mean

import glob, os

os_windows = 0
if os.name == 'nt':  ## in Windows
     os_windows = 1;
     print('Detected Microsoft Windows OS')
else: print('Detected non-Windows OS')


import numpy as np
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

# from functional.plot_functions_CLEANED import *
# from functional.data_functions_CLEANED import *
# from functional.data_functions_3D import *
# from functional.UNet_functions_PYTORCH import *
# from functional.GUI import *


import tifffile as tiff
import z5py

from brainreg.cli import main as brainreg_run
import sys
from postprocess_func import *

import sys
sys.path.append("..")
from get_brain_metadata import *


# list_brains = get_metadate(mouse_num = ['M115', 'M299', 'M138', 'M279', 'M271'])  ### for aging
# list_brains = get_metadata(mouse_num = ['M254', 'M265', 'M312'])    ### P60, cup, and Recovery
# list_brains = get_metadata(mouse_num = ['M147', 'M170'])    ### FVB, CD1



### updated

list_brains = get_metadata(mouse_num = ['M260', 'M286', 'M271', 'Otx6'])  ### for aging
# list_brains = get_metadata(mouse_num = ['M260', 'M265', 'M312'])    ### P60, cup, and Recovery
# list_brains = get_metadata(mouse_num = ['M147'])#'M170'])    ### FVB, CD1





XY_res = 1.152035240378141
Z_res = 5


cloudreg = 0
ANTS = 1

#%% Parse the json file so we can choose what we want to extract or mask out
with open('../atlas_ids/atlas_ids.json') as json_file:
    data = json.load(json_file)
data = data['msg'][0]


# reference_atlas = '/home/user/.brainglobe/633_perens_lsfm_mouse_20um_v1.0/annotation.tiff'

reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation.tiff'

# reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation_10.nrrd'

ref_atlas = tiff.imread(reference_atlas)
# ref_atlas = np.asarray(ref_atlas, dtype=np.uint32)
right_ref_atlas = np.copy(ref_atlas)
right_ref_atlas[:, :, 0:int(ref_atlas.shape[-1]/2)] = 0



ref_boundaries = '/home/user/.brainglobe/allen_mouse_20um_v1.2/boundaries.tif'
ref_bounds = tiff.imread(ref_boundaries)
ref_bounds = np.asarray(ref_bounds, dtype=np.uint32)


#%% Also extract isotropic volumes from reference atlas DIRECTLY - to avoid weird expansion factors 
print('Extracting main key volumes')
keys_dict = get_ids_all(data, all_keys=[], keywords=[''])  
main_keys = pd.DataFrame.from_dict(keys_dict)
main_keys = get_atlas_isotropic_vols(main_keys, ref_atlas, atlas_side='_W', XY_res=20, Z_res=20)
main_keys['atlas_vol_R'] = main_keys['atlas_vol_W']/2
main_keys['atlas_vol_L'] = main_keys['atlas_vol_W']/2

keys_df = main_keys.copy(deep=True)


sav_fold = '/media/user/8TB_HDD/Plot_SLICES/'

density_fold = '/media/user/8TB_HDD/Mean_autofluor/'

for fold in list_brains:
    input_path = fold['path']
    name_str = fold['name']
    
    exp_name = fold['exp']
    if exp_name == 'Cuprizone':
        exp_name = 'CUPRIZONE'
    
    if exp_name == 'Recovery':
        exp_name = 'RECOVERY'
        
        
    if exp_name == 'FVB':
        exp_name = 'FVB'
   
    
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
    
    
    
    # density_path = input_path + name_str + '_postprocess/' + name_str + '_DENSITY_MAP.tif'
    
    density_path = density_fold + exp_name + '_rad5_mean_density_MAP.tif'
    
    

    #%% Get atlas in ORIGINAL orientation and overlay the density map
    density_map = tiff.imread(density_path)
    
    
    def save_2D_slice_low_res(density_map, ref_bounds, ref_atlas, keys_df, reg_name, sav_dir, exp_name, vmin, vmax, axis=0):
        
        if axis==0: ax1=1; ax2=2;
        if axis==1: ax1=0; ax2=2;
        if axis==2: ax1=0; ax2=1;
        
        
        cc = regionprops(ref_atlas, cache=False)
        cc_labs = [region['label'] for region in cc]
        cc_labs = np.asarray(cc_labs)
        
        sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name=reg_name)
        
        sub_keys = keys_df.iloc[sub_idx]
        sub_ids = np.asarray(keys_df['ids'][sub_idx])
        
        
        tmp_atlas_COLOR = np.zeros(np.shape(ref_atlas))
        reg_coords = []
        for idx in sub_ids:
            cur_id = np.where(cc_labs == idx)[0]
            
            #print(cur_id)
            if len(cur_id) == 0:  ### if it does not exists in atlasformation
                continue
            cur_coords = cc[cur_id[0]]['coords']
            tmp_atlas_COLOR[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = idx
            reg_coords.append(cur_coords)
        
        reg_coords = np.vstack(reg_coords)
        
        mid_slice = int(np.mean(reg_coords[:, axis]))
        
        
        if axis == 0 and reg_name == 'Hippocampal region':
            mid_slice = 359
            print('Preset region')
        
        
        
        id_slice = np.where(reg_coords[:, axis] == mid_slice)[0]
        coords_slice = reg_coords[id_slice]
        
        bbox = [np.min(reg_coords[id_slice][:, ax1]), np.min(reg_coords[id_slice][:, ax2]), np.max(reg_coords[id_slice][:, ax1]), np.max(reg_coords[id_slice][:, ax2])]
        
        
        ### plot stuff
        plt.figure(); plt.imshow(tmp_atlas_COLOR[mid_slice],
                                 vmin=vmin, vmax=2000)
        # plot_max(tmp_atlas_COLOR)
        tiff.imwrite(sav_dir + exp_name + '_' + reg_name + '_full_atlas_slice_' + str(mid_slice) + '_AXIS_'  + str(axis) + '.tif', tmp_atlas_COLOR[mid_slice])
              
        
        plt.figure(); plt.imshow(density_map[mid_slice], cmap='turbo', 
                                 vmin=vmin, vmax=vmax)
        # plt.colorbar()
        plt.axis('off');
        plt.savefig(sav_dir + exp_name + '_' + reg_name + '_DENSITY_MAP_FULL_SLICE_' + str(mid_slice) + '_AXIS_'  + str(axis) + '.png', dpi=300, bbox_inches='tight')
        
                
        # Get the cropped bbox areas
        
        if axis==0:
            slice_im = density_map[mid_slice, bbox[0]:bbox[2], bbox[1]:bbox[3]] 
            ### also upscale atlas to allow masking out
            tmp_atlas_COLOR = np.asarray(tmp_atlas_COLOR, dtype=np.uint32)
            crop_down = tmp_atlas_COLOR[mid_slice, bbox[0]:bbox[2], bbox[1]:bbox[3]]
            crop_bounds = ref_bounds[mid_slice, bbox[0]:bbox[2], bbox[1]:bbox[3]]
        
        if axis==1:
            slice_im = density_map[bbox[0]:bbox[2], mid_slice, bbox[1]:bbox[3]] 
            
            # ### need tp scale to isotropic
            # slice_im = rescale(slice_im, [down_factor[1]/down_factor[0], 1], order=1, preserve_range=True)
       
            ### also upscale atlas to allow masking out
            tmp_atlas_COLOR = np.asarray(tmp_atlas_COLOR, dtype=np.uint32)
            crop_down = tmp_atlas_COLOR[bbox[0]:bbox[2], mid_slice, bbox[1]:bbox[3]]
            crop_bounds = ref_bounds[bbox[0]:bbox[2], mid_slice, bbox[1]:bbox[3]]
            
        if axis==2:
            slice_im = density_map[bbox[0]:bbox[2], bbox[1]:bbox[3], mid_slice] 
            
            # ### need tp scale to isotropic
            # slice_im = rescale(slice_im, [down_factor[2]/down_factor[0], 1], order=1, preserve_range=True)
            
            ### also upscale atlas to allow masking out
            tmp_atlas_COLOR = np.asarray(tmp_atlas_COLOR, dtype=np.uint32)
            crop_down = tmp_atlas_COLOR[bbox[0]:bbox[2], bbox[1]:bbox[3], mid_slice]
            crop_bounds = ref_bounds[bbox[0]:bbox[2], bbox[1]:bbox[3], mid_slice]
        
        plt.figure(); plt.imshow(slice_im, cmap='turbo', 
                                 vmin=vmin, vmax=vmax)
        plt.axis('off');
        plt.savefig(sav_dir + exp_name + '_' + reg_name + '_DENSITY_MAP_2D_slice_' + str(mid_slice) + '_AXIS_'  + str(axis) + '.png', dpi=300, bbox_inches='tight')
        tiff.imwrite(sav_dir + exp_name + '_' + reg_name + '_DENSITY_MAP_2D_slice_' + str(mid_slice) + '_AXIS_'  + str(axis) + '.tif', slice_im)
               
        
        plt.figure(); plt.imshow(crop_down,
                                 vmin=vmin, vmax=2000)
        plt.axis('off');
        # plt.savefig(sav_dir + exp_name + '_' + reg_name + '_DENSITY_MAP_2D_slice_ATLAS_' + str(mid_slice) + '_AXIS_'  + str(axis) + '.png', dpi=300, bbox_inches='tight')
        tiff.imwrite(sav_dir + exp_name + '_' + reg_name + '_DENSITY_MAP_2D_slice_ATLAS_' + str(mid_slice) + '_AXIS_'  + str(axis) + '.tif', crop_down)
        
        
        
        plt.figure(); plt.imshow(crop_bounds, cmap='Greys_r')
        plt.axis('off');
        # plt.savefig(sav_dir + exp_name + '_' + reg_name + '_DENSITY_MAP_2D_slice_BOUNDS_' + str(mid_slice) + '_AXIS_'  + str(axis) + '.png', dpi=300, bbox_inches='tight')
        tiff.imwrite(sav_dir + exp_name + '_' + reg_name + '_DENSITY_MAP_2D_slice_BOUNDS_' + str(mid_slice) + '_AXIS_'  + str(axis) + '.tif', crop_bounds)
                      

        
        
    
        
    # save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
    #                        reg_name='Primary somatosensory area', exp_name=fold['name'],
    #                         sav_dir=sav_fold, axis=0, vmin=0, vmax=25000)
    
    # save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
    #                        reg_name='Retrosplenial area', exp_name=fold['name'],
    #                         sav_dir=sav_fold, axis=0, vmin=0, vmax=30000)

    # save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
    #                        reg_name='Entorhinal area', exp_name=fold['name'],
    #                         sav_dir=sav_fold, axis=0, vmin=0, vmax=30000)

    # # save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
    #                        reg_name='Hippocampal formation', exp_name=fold['name'],
    #                         sav_dir=sav_fold, axis=0, vmin=0, vmax=30000)


    save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
                           reg_name='Hippocampal region', exp_name=fold['name'],
                            sav_dir=sav_fold, axis=0, vmin=0, vmax=30000)
    
    

    save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
                            reg_name='Thalamus', exp_name=fold['name'],
                            sav_dir=sav_fold, axis=0, vmin=0, vmax=30000)
    
    # # save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
    # #                        reg_name='Striatum', exp_name=fold['name'],
    # #                         sav_dir=sav_fold, axis=0, vmin=0, vmax=30000)
    
    # # save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
    # #                        reg_name='fiber tracts', exp_name=fold['name'],
    # #                         sav_dir=sav_fold, axis=0, vmin=0, vmax=30000)
    
    # # save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
    # #                        reg_name='Midbrain', exp_name=fold['name'],
    # #                         sav_dir=sav_fold, axis=0, vmin=0, vmax=30000)
    
    # # save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
    # #                        reg_name='Hypothalamus', exp_name=fold['name'],
    # #                         sav_dir=sav_fold, axis=0, vmin=0, vmax=30000)
    
    save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
                            reg_name='Cerebellar cortex', exp_name=fold['name'],
                            sav_dir=sav_fold, axis=0, vmin=0, vmax=30000)
            
    



    
    #%% Get registered atlas and shift the axes
    myelin = tiff.imread(myelin_path)
    
    # atlas = tiff.imread(atlas_dir + '/registered_atlas.tiff')
    
    
    ## WHOLE ATLAS COMBINED --- for Cerebellum
    atlas = tiff.imread(atlas_dir + '/atlas_combined_CEREBELLUM_AND_IC.tif')
    
    
    
    
    
    
    atlas = np.moveaxis(atlas, 0, 1)   ### reshuffle atlas so now in proper orientation
    atlas = np.flip(atlas, axis=0)  ### flip the Z-axis
    atlas = np.flip(atlas, axis=2)
    atlas_size_pre_resize = atlas.shape
    atlas = resize(atlas, myelin.shape, anti_aliasing=False, order=0, preserve_range=True)   ### rescale the images
    
    
    hemispheres = tiff.imread(atlas_dir + '/registered_hemispheres.tiff')
    hemispheres = np.moveaxis(hemispheres, 0, 1)   ### reshuffle atlas so now in proper orientation
    hemispheres = np.flip(hemispheres, axis=0)  ### flip the Z-axis
    hemispheres = np.flip(hemispheres, axis=2)  ### flip the Z-axis
    
    hemispheres = resize(hemispheres, myelin.shape, anti_aliasing=False, order=0, preserve_range=True)   ### rescale the images
        
    right_atlas = np.copy(atlas)
    right_atlas[hemispheres == 1] = 0
        
    # left_atlas = np.copy(atlas)
    # left_atlas[hemispheres == 2] = 0
    
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
    #atlas_nopad = atlas[z_after:-z_after, x_after:-x_after, x_after:-x_after]
    right_nopad = right_atlas[z_after:-z_after, x_after:-x_after, x_after:-x_after]
    # left_nopad = left_atlas[z_after:-z_after, x_after:-x_after, x_after:-x_after]
    
        
    
    
    #%% Extract region of interest HIGH RES
    
    def save_2D_slice_high_res(dset, atlas, keys_df, reg_name, sav_dir, exp_name, down_factor, axis=0):
        
        if axis==0: ax1=1; ax2=2;
        if axis==1: ax1=0; ax2=2;
        if axis==2: ax1=0; ax2=1;
        
        
        cc = regionprops(atlas, cache=False)
        cc_labs = [region['label'] for region in cc]
        cc_labs = np.asarray(cc_labs)
        
        sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name=reg_name)
        
        
        sub_keys = keys_df.iloc[sub_idx]
        sub_ids = np.asarray(keys_df['ids'][sub_idx])
        
        
        tmp_atlas_COLOR = np.zeros(np.shape(atlas))
        reg_coords = []
        for idx in sub_ids:
            cur_id = np.where(cc_labs == idx)[0]
            
            #print(cur_id)
            if len(cur_id) == 0:  ### if it does not exists in atlas
                continue
            cur_coords = cc[cur_id[0]]['coords']
            tmp_atlas_COLOR[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = idx
            reg_coords.append(cur_coords)
        
        reg_coords = np.vstack(reg_coords)
        
        
                

        
        mid_slice = int(np.mean(reg_coords[:, axis]))
        
        print(mid_slice)
        # zzz
        if axis == 1 and reg_name == 'Hippocampal region':
            mid_slice = 359
            # mid_slice = 300
            print('Preset region')
        
        
        
        id_slice = np.where(reg_coords[:, axis] == mid_slice)[0]
        coords_slice = reg_coords[id_slice]
        
        bbox_down = [np.min(reg_coords[id_slice][:, ax1]), np.min(reg_coords[id_slice][:, ax2]), np.max(reg_coords[id_slice][:, ax1]), np.max(reg_coords[id_slice][:, ax2])]
        
        

        
        #%% Then get 2D slice by using bounding box
        
        # GET THE HIGH RES SLICE coordinates
        bbox = np.copy(bbox_down)
        bbox[0] = bbox[0] * down_factor[ax1]; bbox[2] = bbox[2] * down_factor[ax1]
        bbox[1] = bbox[1] * down_factor[ax2]; bbox[3] = bbox[3] * down_factor[ax2]
        bbox = np.asarray(bbox, dtype=np.int32)
        mid = int(mid_slice * down_factor[axis])
        
        if axis==0:
            slice_im = dset[mid, bbox[0]:bbox[2], bbox[1]:bbox[3]] 
            ### also upscale atlas to allow masking out
            tmp_atlas_COLOR = np.asarray(tmp_atlas_COLOR, dtype=np.uint32)
            crop_down = tmp_atlas_COLOR[mid_slice, bbox_down[0]:bbox_down[2], bbox_down[1]:bbox_down[3]]
            
            atlas_slice = atlas[mid_slice, bbox_down[0]:bbox_down[2], bbox_down[1]:bbox_down[3]] 
            
        
        if axis==1:
            slice_im = dset[bbox[0]:bbox[2], mid, bbox[1]:bbox[3]] 
            
            ### need tp scale to isotropic
            slice_im = rescale(slice_im, [down_factor[1]/down_factor[0], 1], order=1, preserve_range=True)
       
            ### also upscale atlas to allow masking out
            tmp_atlas_COLOR = np.asarray(tmp_atlas_COLOR, dtype=np.uint32)
            crop_down = tmp_atlas_COLOR[bbox_down[0]:bbox_down[2], mid_slice, bbox_down[1]:bbox_down[3]]
            
            ### need to flip
            slice_im = np.flip(slice_im, axis=0)
            slice_im = np.asarray(slice_im, dtype=np.uint16)
            
            crop_down = np.flip(crop_down, axis=0)
            
            
            atlas_slice = atlas[bbox_down[0]:bbox_down[2], mid_slice, bbox_down[1]:bbox_down[3]] 
            atlas_slice = np.flip(atlas_slice, axis=0)
            
            
        if axis==2:
            slice_im = dset[bbox[0]:bbox[2], bbox[1]:bbox[3], mid] 
            
            ### need tp scale to isotropic
            slice_im = rescale(slice_im, [down_factor[2]/down_factor[0], 1], order=1, preserve_range=True)
            
            ### also upscale atlas to allow masking out
            tmp_atlas_COLOR = np.asarray(tmp_atlas_COLOR, dtype=np.uint32)
            crop_down = tmp_atlas_COLOR[bbox_down[0]:bbox_down[2], bbox_down[1]:bbox_down[3], mid_slice]
        
        
        mask_rescaled = resize(crop_down, slice_im.shape, order=0)
        
        tiff.imwrite(sav_dir + exp_name + '_' + reg_name + '_EXTRACTED_2D_slice_' + str(mid) + '_AXIS_'  + str(axis) + '.tif', slice_im)
        tiff.imwrite(sav_dir + exp_name + '_' + reg_name + '_EXTRACTED_2D_ATLAS_' + str(mid) + '_AXIS_'  + str(axis) + '.tif', np.asarray(mask_rescaled, dtype=np.uint32))
        
        
        ### also mask out surroundings
        # slice_im[mask_rescaled == 0] = 0
        tiff.imwrite(sav_dir + exp_name + '_' + reg_name + '_EXTRACTED_2D_MASKED_' + str(mid) + '_AXIS_'  + str(axis) + '.tif', slice_im)
    
    
        from skimage.segmentation import find_boundaries
        # plot_max(atlas_slice)
        bw_slice = find_boundaries(atlas_slice)
        bw_slice = np.asarray(bw_slice, dtype=np.uint8)
        bw_slice[bw_slice > 0] = 1
        
        bw_slice = resize(bw_slice, slice_im.shape, order=0)         
        tiff.imwrite(sav_dir + exp_name + '_' + reg_name + '_EXTRACTED_2D_BOUNDS_lowres_' + str(mid) + '_AXIS_'  + str(axis) + '.tif', np.asarray(bw_slice, dtype=np.uint8))
        
        
    
    
    
    # # # Done
    # # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Primary somatosensory area', 
    # #                         sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=0)
    
    # # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Primary somatosensory area', 
    # #                         sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=1)
    


    save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Thalamus', 
                            sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=1)
    

    save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Cerebellar cortex', 
                            sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=1)
    
    # # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Retrosplenial area', 
    # #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=0)
    
    # # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Retrosplenial area', 
    # #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=1)


    # # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Ectorhinal area', 
    # #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=0)
    
    # # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Ectorhinal area', 
    # #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=1)
    
    
    
    # # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Entorhinal area', 
    # #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=0)
    
    # # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Entorhinal area', 
    # #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=1)
    
    
    
    save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Hippocampal region', 
                           sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=1)
    
    # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Striatum', 
    #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=0)
    
    # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Striatum', 
    #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=1)


    # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Midbrain', 
    #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=0)
    
    # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Midbrain', 
    #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=1)


    # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Hypothalamus', 
    #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=0)
    
    # save_2D_slice_high_res(dset, atlas=right_nopad, keys_df=keys_df, reg_name='Hypothalamus', 
    #                        sav_dir=sav_fold, exp_name=fold['name'], down_factor=down_factor, axis=1)
                                

    
    # save_2D_slice_low_res(density_map, ref_bounds, ref_atlas=right_ref_atlas, keys_df=keys_df, 
    #                        reg_name='fiber tracts', exp_name=fold['name'],
    #                         sav_dir=sav_fold, axis=0, vmin=0, vmax=30000)

    



    
    
    
        