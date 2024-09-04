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



#%% List of brains
# list_brains = get_metadata(mouse_num = 'all')

# list_brains = get_metadata(mouse_num = ['M97'])
# list_brains = get_metadata(mouse_num = ['Otx18'])
# # list_brains = get_metadata(mouse_num = ['M310', 'M312'])


# # list_brains = get_metadata(mouse_num = ['M91', 'M271', 'M334'])


# list_brains = get_metadata(mouse_num = ['5Otx5', 'Otx6'])

# list_brains = get_metadata(mouse_num = ['M265', 'M266', 'M267'])   ### dont bother using full autofluor brain, doesnt get rid of cerebellum as nicely


# list_brains = get_metadata(mouse_num = ['M267', 'M147'])


# list_brains = get_metadata(mouse_num = ['M169'])


# list_brains = get_metadata(mouse_num = ['M310', 'M313', 'M312', 'M265', 'M266', 'M267'])

# list_brains = get_metadata(mouse_num = ['M265', 'M266', 'M267', 'M310', 'M311', 'M312', 'M313', 'M248', 'M246']) 

# # FVB/CD1
# list_brains = get_metadata(mouse_num = ['M147', 'M155', 'M152', 'M170', 'M172']) 


# P60
# list_brains = get_metadata(mouse_num = ['M127', 'M229', 'M126', 'M299', 'M254', 'M256', 'M260', 'M223'])




### Re-run
# list_brains = get_metadata(mouse_num = ['M334', 'M271'])


# list_brains = get_metadata(mouse_num = ['M286', 'M256'])

list_brains = get_metadata(mouse_num = ['M126', 'M127'])

# list_brains = get_metadata(mouse_num = ['M246'])


cloudreg = 0

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
  
    # actual everything, no gauss, -20 GRID
    # atlas_dir = downsampled_dir + name_str + '633_perens_lsfm_mouse_20um_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n40.9_n4_1_grid_-20_gauss_0_use_steps_default_PADDED_50/'

    # actual everything, no gauss, -15 GRID
    # atlas_dir = downsampled_dir + name_str + '633_perens_lsfm_mouse_20um_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n40.9_n4_1_grid_-15_gauss_0_use_steps_default_PADDED_50/'

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

    ### if want to forcefully use CUBIC brain average template --- used for cuprizone + cuprizone recovery
    # atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_20um_CUBIC_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'





    scaled = 1





    allen_dir = downsampled_dir + name_str + '_ISOCORTEX_CORTEX_ONLY_allen_mouse_10um_bend_0.95_grid_-10_gauss_0/'
    analysis_dir = input_path + name_str + '_MaskRCNN_patches/'
    

    myelin_path = glob.glob(os.path.join(downsampled_dir,'*_ch0_n4_down1_resolution_20_PAD.tif'))[0]    # can switch this to "*truth.tif" if there is no name for "input"
    auto_path = glob.glob(os.path.join(downsampled_dir, '*_ch1_n4_down1_resolution_20_PAD.tif'))[0]
    
 
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
    

    
    if not os.path.isfile(sav_dir + filename + '_coords_TMP.pkl'):
        # load all pickle files and parse into larger df
        all_pkl = []
        coords_only = []
        #vols_only = []
        

        for blk_num in range(len(examples)):
         
            #try:
            pkl = pd.read_pickle(examples[blk_num]['pkl'])
            
            if len(pkl) == 0:
                print('No cells in pickle?')
                continue
            # except:
            #     print(examples[blk_num]['pkl'])
            #     print(blk_num)
            # also get volume
    
            vols = [len(coords) for coords in pkl['coords_scaled']]
            #vols_only = np.concatenate((vols_only, vols))
            
            ### scale coords up to isovolumetric and then to microns
            # vols = []
            # for coords in pkl['coords_scaled']:
            #     vol = len(coords)
            #     vol = vol * 1/res_diff  ### scale to volumetric
            #     vol = vol / XY_res      ### scale to um^3
                
            #     vols.append(vol)
            
            # if want ALL the info --- currently missing equivalent diameter??? Might be easier to just do in 2D for volume comparisons...
            # saves RAM if removed
            #all_pkl.append(pkl)
            
            # if want just coords
            c_pkl = pkl[['X_scaled', 'Y_scaled', 'Z_scaled']].copy()
            c_pkl['vols'] = vols
            
            coords_only.append(c_pkl)
           
            print(blk_num)
        
        
        coords_df = pd.concat(coords_only)
        coords_df.reset_index(inplace=True, drop=True)
        
        ### scale the volumes
        
        
        
        coords_df['vols_um'] = coords_df['vols'] * 1/res_diff   ### scale to isotropic
        coords_df['vols_um'] = coords_df['vols_um'] / XY_res
        
        
        overlap_pxy = pkl['overlap_pxy'][0]
        overlap_pz = pkl['overlap_pz'][0]
    
        #%% SAVE PICKLE SO DONT HAVE TO REDO THIS CONSTANTLY IN THE FUTURE
        coords_df.to_pickle(sav_dir + filename + '_coords_TMP.pkl')
    
        
    else:
        ### if already done this before, just re-open
        coords_df = pd.read_pickle(sav_dir + filename + '_coords_TMP.pkl')
        pkl = pd.read_pickle(examples[10]['pkl'])
        overlap_pxy = pkl['overlap_pxy'][0]
        overlap_pz = pkl['overlap_pz'][0]
        
    

    size = 30   ### pixels
    ### APPLY SIZE THRESHOLD
    coords_df = coords_df.iloc[np.where(coords_df['vols'] > size)[0]]
    
    

    #%% Load
    
    #autofluor = tiff.memmap(auto_path)
    myelin = tiff.memmap(myelin_path)
    

    #%% Load registered atlas
    print('Loading registered atlas')
    #dirlist = os.listdir(input_path)
    #atlas_dir = [s for s in dirlist if 'allen_mouse_25um' in s][0]
    
    
    # zzz
    #%% Also load the WHOLE_BRAIN registered --- so can extract cerebellum + IC
    if os.path.exists(atlas_dir + '/atlas_combined_CEREBELLUM_AND_IC.tif'):
        atlas = tiff.imread(atlas_dir + '/atlas_combined_CEREBELLUM_AND_IC.tif')
        
        
    else:
        atlas = tiff.imread(atlas_dir + '/registered_atlas.tiff')
    
    
    hemispheres = tiff.imread(atlas_dir + '/registered_hemispheres.tiff')
    right_atlas = np.copy(atlas)
    right_atlas[hemispheres == 1] = 0

    left_atlas = np.copy(atlas)
    left_atlas[hemispheres == 2] = 0

    if  not os.path.exists(atlas_dir + '/atlas_combined_CEREBELLUM_AND_IC.tif'):    
        # get atlas including cerebellum + IC reg
        atlas_WHOLE = tiff.imread(atlas_WHOLE_dir + '/registered_atlas.tiff')
        hemispheres_WHOLE = tiff.imread(atlas_WHOLE_dir + '/registered_hemispheres.tiff')
        right_WHOLE = np.copy(atlas_WHOLE)
        right_WHOLE[hemispheres_WHOLE == 1] = 0
            
        left_WHOLE = np.copy(atlas_WHOLE)
        left_WHOLE[hemispheres_WHOLE == 2] = 0
        
        
        
        ### First start by defining everything we do NOT want
        all_sub_id = []
        sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Cerebellum')
        all_sub_id.append(sub_idx)
        sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='cerebellum related fiber tracts')
        all_sub_id.append(sub_idx)
        sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Inferior colliculus')
        all_sub_id.append(sub_idx)
        
        all_sub_id = [x for xs in all_sub_id for x in xs]
        
        sub_keys = keys_df.iloc[all_sub_id]
        sub_keys.reset_index(inplace=True, drop=True)
        
        
        sub_ids = np.asarray(sub_keys['ids'])
        
        ### remove additional minor tracts
        for val in [326, 78, 866, 812, 553, 1123]: # (512 needs to be reduced but it goes everywhere...)
            sub_ids = sub_ids[sub_ids != val]
    
    
        delR = delL = coordsR = coordsL = []
        remove_regions = np.zeros(np.shape(atlas_WHOLE))
        for idx in sub_ids:
            print(idx)
            delR.append(np.transpose(np.where(right_atlas == idx)))
            delL.append(np.transpose(np.where(left_atlas == idx)))
            
            ### Then add in new ones
            coordsR.append(np.transpose(np.where(right_WHOLE == idx)))
            coordsL.append(np.transpose(np.where(left_WHOLE == idx))) 
            
        # add all coords together
        fullR = np.vstack(delR + coordsR)
        fullL = np.vstack(delL + coordsL)
        
        right_atlas[fullR[:, 0], fullR[:, 1], fullR[:, 2]] = right_WHOLE[fullR[:, 0], fullR[:, 1], fullR[:, 2]]
        left_atlas[fullL[:, 0], fullL[:, 1], fullL[:, 2]] = left_WHOLE[fullL[:, 0], fullL[:, 1], fullL[:, 2]]
        
        
    
        
        #%% Also update deformation fields to have update cerebellum + IC
        fullW = np.concatenate([fullR, fullL])
        
        deformation_field_paths = [atlas_dir + 'deformation_field_0.tiff',
                                    atlas_dir + 'deformation_field_1.tiff',
                                    atlas_dir + 'deformation_field_2.tiff'
                                  ]
        
        deformation_field_WHOLE = [atlas_WHOLE_dir + 'deformation_field_0.tiff',
                                    atlas_WHOLE_dir + 'deformation_field_1.tiff',
                                    atlas_WHOLE_dir + 'deformation_field_2.tiff'
                                  ]
    
    
        for id_f, field_p in enumerate(deformation_field_paths):
            print('Hello')
            field = tiff.imread(field_p)
            field_WHOLE = tiff.imread(deformation_field_WHOLE[id_f])
            field[fullW[:, 0], fullW[:, 1], fullW[:, 2]] = field_WHOLE[fullW[:, 0], fullW[:, 1], fullW[:, 2]]
            
            tiff.imwrite(field_p[:-5] + '_ADD_CERE.tiff', field)
    
    
        
        
        clean_atlas = np.zeros(np.shape(atlas), dtype=np.uint32)
        clean_atlas[right_atlas > 0] = right_atlas[right_atlas > 0] 
        clean_atlas[left_atlas > 0] = left_atlas[left_atlas > 0]
    
        tiff.imwrite(atlas_dir + '/atlas_combined_CEREBELLUM_AND_IC.tif', clean_atlas)
    
    
        


    #%% Flip atlas to correct orientation    
    
    right_atlas = np.moveaxis(right_atlas, 0, 1)   ### reshuffle atlas so now in proper orientation
    right_atlas = np.flip(right_atlas, axis=0)  ### flip the Z-axis
    right_atlas = np.flip(right_atlas, axis=2)
    atlas_size_pre_resize = atlas.shape
    right_atlas = resize(right_atlas, myelin.shape, anti_aliasing=False, order=0, preserve_range=True)   ### rescale the images
    
    
    left_atlas = np.moveaxis(left_atlas, 0, 1)   ### reshuffle atlas so now in proper orientation
    left_atlas = np.flip(left_atlas, axis=0)  ### flip the Z-axis
    left_atlas = np.flip(left_atlas, axis=2)
    atlas_size_pre_resize = atlas.shape
    left_atlas = resize(left_atlas, myelin.shape, anti_aliasing=False, order=0, preserve_range=True)   ### rescale the images
    
    # get whole atlas
    atlas = np.zeros(np.shape(right_atlas), dtype=np.uint32)
    atlas[right_atlas > 0] = right_atlas[right_atlas > 0] 
    atlas[left_atlas > 0] = left_atlas[left_atlas > 0]



    # atlas = np.moveaxis(atlas, 0, 1)   ### reshuffle atlas so now in proper orientation
    # atlas = np.flip(atlas, axis=0)  ### flip the Z-axis
    # atlas = np.flip(atlas, axis=2)
    # atlas_size_pre_resize = atlas.shape
    # atlas = resize(atlas, myelin.shape, anti_aliasing=False, order=0, preserve_range=True)   ### rescale the images
    
 
    # hemispheres = np.moveaxis(hemispheres, 0, 1)   ### reshuffle atlas so now in proper orientation
    # hemispheres = np.flip(hemispheres, axis=0)  ### flip the Z-axis
    # hemispheres = np.flip(hemispheres, axis=2)  ### flip the Z-axis
    # hemispheres = resize(hemispheres, myelin.shape, anti_aliasing=False, order=0, preserve_range=True)   ### rescale the images



        
    # right_atlas = np.copy(atlas)
    # right_atlas[hemispheres == 1] = 0
        
    # left_atlas = np.copy(atlas)
    # left_atlas[hemispheres == 2] = 0
    
    
    
    
    tiff.imwrite(sav_dir + 'left_atlas.tif',  np.expand_dims(np.expand_dims(np.asarray(left_atlas, dtype=np.uint32), axis=0), axis=2),)
                 #imagej=True, resolution=(1/XY_res, 1/XY_res),
                 #metadata={'spacing':Z_res, 'unit': 'um', 'axes': 'TZCYX'})
        
    tiff.imwrite(sav_dir + 'right_atlas.tif',  np.expand_dims(np.expand_dims(np.asarray(right_atlas, dtype=np.uint32), axis=0), axis=2),)
                 #imagej=True, resolution=(1/XY_res, 1/XY_res),
                 #metadata={'spacing':Z_res, 'unit': 'um', 'axes': 'TZCYX'})
    
    
    tiff.imwrite(sav_dir + 'whole_atlas.tif',  np.expand_dims(np.expand_dims(np.asarray(atlas, dtype=np.uint32), axis=0), axis=2),)
                 #imagej=True, resolution=(1/XY_res, 1/XY_res),
                 #metadata={'spacing':Z_res, 'unit': 'um', 'axes': 'TZCYX'})
        
    ### garbage collect:
    #left_atlas = []
    #right_atlas = []
    hemispheres = []
    
    

    #%% Only do gray matter comparisons to start
    print('Converting coordinates to correct scale')
    ### supposedly atlass is at a 16 fold downsample in XY and no downsampling in Z
    
    """ If padded earlier, add padding here
    """

    ### Load n5 file
    #with z5py.File(input_name, "r") as f:
    
    f = z5py.File(n5_file, "r")
    dset = f['setup0/timepoint0/s0']    
     
    
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
    print('myelin.shape')
     
    
       
    coords_df['Z_down'] = (coords_df['Z_scaled'] - overlap_pz)/down_factor[0]
    coords_df['X_down'] = (coords_df['X_scaled'] - overlap_pxy)/down_factor[1]
    coords_df['Y_down'] = (coords_df['Y_scaled'] - overlap_pxy)/down_factor[2]
    
    
    if pad:
        # scale coords with padding if originally had padding for registration
        coords_df['Z_down'] = coords_df['Z_down'] + z_after
        coords_df['X_down'] = coords_df['X_down'] + x_after
        coords_df['Y_down'] = coords_df['Y_down'] + x_after
    
    
    cell_pos = [coords_df['Z_down'].astype(int), coords_df['X_down'].astype(int), coords_df['Y_down'].astype(int)]
    cell_pos = np.transpose(np.asarray(cell_pos))
    
    ### make sure coordinates stay in range
    cell_pos[np.where(cell_pos[:, 0] >= atlas.shape[0])[0], 0] = atlas.shape[0] - 1
    cell_pos[np.where(cell_pos[:, 1] >= atlas.shape[1])[0], 1] = atlas.shape[1] - 1
    cell_pos[np.where(cell_pos[:, 2] >= atlas.shape[2])[0], 2] = atlas.shape[2] - 1 
    print('CHECKING COORDS OUT OF FOV')
    
    
    
    
    #%% Then map cell positions and densities to atlas locations
    ### INCLUDES ISOTROPIC SCALING TO ATLAS SIZES
    
    # zzz
    # print('saving COLORED cells image')
    # # make random values
    # random_array = np.random.randint(0, 255, len(cell_pos))

    
    # cells_im_COLOR = np.zeros(np.shape(atlas))
    # cells_im_COLOR[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]] = random_array
    # # tiff.imwrite(sav_dir + 'cells_im_COLOR.tif', np.asarray(cells_im_COLOR, dtype=np.uint8))    
    
    # ### Add dilation by ball
    # from skimage.morphology import ball, disk, dilation
    # """ dilates image by a spherical ball of size radius """
    # def dilate_by_ball_to_grayscale(input_im, radius):
    #       ball_obj = ball(radius=radius)
    #       input_im = dilation(input_im, footprint=ball_obj)  
    #       #input_im[input_im > 0] = 1
    #       return input_im
      
    # cells_im_COLOR_dil = dilate_by_ball_to_grayscale(cells_im_COLOR, radius=1)
    # tiff.imwrite(sav_dir + 'cells_im_COLOR_dil.tif', np.asarray(cells_im_COLOR_dil, dtype=np.uint8))    
       
                
    
    
    
    
    print('saving cells image')
    cells_im = np.zeros(np.shape(atlas))
    cells_im[cell_pos[:, 0], cell_pos[:, 1], cell_pos[:, 2]] = 255
    tiff.imwrite(sav_dir + 'cells_im.tif', np.asarray(cells_im, dtype=np.uint8))    
    
    
    if not cloudreg:
        keys_df = cells_to_atlas_df(keys_df, coords_df, cell_pos=cell_pos, atlas=left_atlas, atlas_side='_L', 
                                    XY_res=XY_res * down_factor[-1], Z_res=Z_res * down_factor[0], size_thresh=500)
        
        left_atlas = []   # garbage collect
        
        keys_df = cells_to_atlas_df(keys_df, coords_df, cell_pos=cell_pos, atlas=right_atlas, atlas_side='_R', 
                                    XY_res=XY_res * down_factor[-1], Z_res=Z_res * down_factor[0], size_thresh=500)
        
        right_atlas = []  # garbage collect
    
    keys_df = cells_to_atlas_df(keys_df, coords_df, cell_pos=cell_pos, atlas=atlas, atlas_side='_W', 
                                XY_res=XY_res * down_factor[-1], Z_res=Z_res * down_factor[0], size_thresh=500)




    #%% SAVE keys_df and cell position df for multi-brain comparisons AND for size comparisons
    # keys_df.to_pickle(sav_dir + filename + '_keys_df_PERENS_EVERYTHING-15grid.pkl')
    # coords_df.to_pickle(sav_dir + filename + '_coords_df_PERENS_EVERYTHING-15grid.pkl')

    # keys_df.to_pickle(sav_dir + filename + '_keys_df_ALLEN_EVERYTHING-15grid.pkl')
    # coords_df.to_pickle(sav_dir + filename + '_coords_df_ALLEN_EVERYTHING-15grid.pkl')

    if not ANTS:
        keys_df.to_pickle(sav_dir + filename + '_keys_df_ALLEN_EVERYTHING-10grid.pkl')
        coords_df.to_pickle(sav_dir + filename + '_coords_df_ALLEN_EVERYTHING-10grid.pkl')
    else:
        keys_df.to_pickle(sav_dir + filename + '_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE.pkl')
        coords_df.to_pickle(sav_dir + filename + '_coords_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE.pkl')        
        
    
    print('EXTRACTED AND SAVED COORDS FROM: ' + name_str)
    
    
    
    #%% 1. start with hemisphere comparisons
    
    #sns.scatterplot(x='density_L', y='density_R', data=keys_df)
    sns.lmplot(x='density_L', y='density_R', data=keys_df.dropna())
    r,p = stats.pearsonr(keys_df.dropna()['density_L'], keys_df.dropna()['density_R'])
    print(r)
    
    sns.lmplot(x='num_OLs_L', y='num_OLs_R', data=keys_df.dropna())
    r,p = stats.pearsonr(keys_df.dropna()['num_OLs_L'], keys_df.dropna()['num_OLs_R'])    
    print(r)
    
    sns.lmplot(x='num_large_L', y='num_large_R', data=keys_df)
    r,p = stats.pearsonr(keys_df.dropna()['num_large_L'], keys_df.dropna()['num_large_R'])    
    print(r) 
    
    ### Find regions that do NOT match
    keys_df['num_OLs_absdiff'] = abs(keys_df['num_OLs_L'] - keys_df['num_OLs_R'])
    keys_df['num_OLs_scaleddiff'] = keys_df['num_OLs_absdiff']/keys_df['num_OLs_W']   


    
