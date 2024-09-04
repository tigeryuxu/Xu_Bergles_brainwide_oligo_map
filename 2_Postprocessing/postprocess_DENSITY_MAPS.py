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
list_brains = get_metadata(mouse_num = 'all')
# list_brains = get_metadata(mouse_num = ['M254']) #'M265', 'M312'])    ### P60, cup, and Recovery
# list_brains = get_metadata(mouse_num = ['M147', 'M170'])    ### FVB, CD1

# list_brains = get_metadata(mouse_num = ['M271'])
# list_brains = get_metadata(mouse_num = ['M254'])

list_brains = get_metadata(mouse_num = ['M299'])


# list_brains = get_metadata(mouse_num = ['M127', 'M229', 'M126', 'M299', 'M254', 'M256', 'M260', 'M223'])


# list_brains = get_metadata(mouse_num = ['M127', 'M265', 'M266', 'M267', 'M310', 'M311', 'M312', 'M313']) 


# # FVB/CD1
# list_brains = get_metadata(mouse_num = ['M147', 'M155', 'M152', 'M170', 'M172']) 



cloudreg = 0
ANTS = 1


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
    pkl = pd.read_pickle(examples[10]['pkl'])
    overlap_pxy = pkl['overlap_pxy'][0]
    overlap_pz = pkl['overlap_pz'][0]
    

    
    
    #%% REMOVE VERY SMALL DETECTIONS
    size = 30   ### pixels
    ### APPLY SIZE THRESHOLD
    coords_df = coords_df.iloc[np.where(coords_df['vols'] > size)[0]]
    
    
    

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
    print('myelin.shape')
     
    
       
    coords_df['Z_down'] = (coords_df['Z_scaled'] - overlap_pz)/down_factor[0]
    coords_df['X_down'] = (coords_df['X_scaled'] - overlap_pxy)/down_factor[1]
    coords_df['Y_down'] = (coords_df['Y_scaled'] - overlap_pxy)/down_factor[2]
    
    
    if pad:
        # scale coords with padding if originally had padding for registration
        coords_df['Z_down'] = coords_df['Z_down'] + z_after
        coords_df['X_down'] = coords_df['X_down'] + x_after
        coords_df['Y_down'] = coords_df['Y_down'] + x_after
    
    
    # cell_pos = [coords_df['Z_down'].astype(int), coords_df['X_down'].astype(int), coords_df['Y_down'].astype(int)]
    # cell_pos = np.transpose(np.asarray(cell_pos))
    
    # ### make sure coordinates stay in range
    # cell_pos[np.where(cell_pos[:, 0] >= atlas.shape[0])[0], 0] = atlas.shape[0] - 1
    # cell_pos[np.where(cell_pos[:, 1] >= atlas.shape[1])[0], 1] = atlas.shape[1] - 1
    # cell_pos[np.where(cell_pos[:, 2] >= atlas.shape[2])[0], 2] = atlas.shape[2] - 1 
    # print('CHECKING COORDS OUT OF FOV')
    

    #%% SET OPTION TO SKIP BELOW IF ALREADY EXTRACTED
    # if not os.path.isfile(sav_dir + exp['name'] + '_DENSITY_MAP.tif'):
    if True:

        
        #%% MAKE DENSITY MAP:
        downsampled_points = np.asarray([coords_df['Z_down'], coords_df['X_down'], coords_df['Y_down']]).T
        # and then move columns (1 to 0)  --> all these steps just do the inverse of original steps
        downsampled_points[:, [1, 0, 2]] = downsampled_points[:, [0, 1, 2]]
        
    
        # deformation_field_paths = [atlas_dir + 'deformation_field_0.tiff',
        #                             atlas_dir + 'deformation_field_1.tiff',
        #                             atlas_dir + 'deformation_field_2.tiff'
        #                           ]

        deformation_field_paths = [atlas_dir + 'deformation_field_0_ADD_CERE.tiff',
                                    atlas_dir + 'deformation_field_1_ADD_CERE.tiff',
                                    atlas_dir + 'deformation_field_2_ADD_CERE.tiff'
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
        
        p = np.copy(points)
            
    
        ### plot onto ref_atlas???
        # points_ref = np.zeros(np.shape(ref_atlas))
        
        # ### num out of bounds:
        # len(np.where(points[:, 0] >= np.shape(points_ref)[0])[0])    ### most of them in this axis --- brainstem area maybe??? or olfactory
        # len(np.where(points[:, 1] >= np.shape(points_ref)[1])[0])
        # len(np.where(points[:, 2] >= np.shape(points_ref)[2])[0])
        
        
        # # remove all out of bounds
        # p[np.where(p[:, 0] >= np.shape(points_ref)[0])[0], 0] = np.shape(points_ref)[0] - 1
        # p[np.where(p[:, 1] >= np.shape(points_ref)[1])[0], 1] = np.shape(points_ref)[1] - 1
        # p[np.where(p[:, 2] >= np.shape(points_ref)[2])[0], 2] = np.shape(points_ref)[2] - 1
          
        
        # points_ref[p[:, 0], p[:, 1], p[:, 2]] = 1
        
        
        # plt.figure(); plt.imshow(points_ref[300])
        # plt.figure(); plt.imshow(ref_atlas[300])
        
    
    
                
        #%%## in 3D
        # import numpy as np
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        print('3D density map')
        
        center = [0, 0, 0]
        radius = 5   ### 20um x 20 pixel diameter == 400 um x 400 um x 400 um
        
        # Generate a meshgrid of coordinates
        x_range = np.arange(center[0] - radius, center[0] + radius + 1)
        y_range = np.arange(center[1] - radius, center[1] + radius + 1)
        z_range = np.arange(center[2] - radius, center[2] + radius + 1)
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # Calculate the distance from each point to the center
        distances = np.sqrt((xx - center[0])**2 + (yy - center[1])**2 + (zz - center[2])**2)
        
        # Create a mask for points inside the sphere
        mask = distances <= radius
        
        # Extract coordinates inside the sphere
        sphere_c = np.column_stack([xx[mask], yy[mask], zz[mask]])
        
        
        ### Now loop through and sum spheres
        sphere_im = np.zeros(np.shape(ref_atlas))
        
        for id_c, cell_center in enumerate(p):
            
            cur_sphere = cell_center + sphere_c   ### make sphere centered around current centroid
            
            
            # remove all out of bounds parts of spheres
            cur_sphere[np.where(cur_sphere[:, 0] >= np.shape(sphere_im)[0])[0], 0] = np.shape(sphere_im)[0] - 1
            cur_sphere[np.where(cur_sphere[:, 1] >= np.shape(sphere_im)[1])[0], 1] = np.shape(sphere_im)[1] - 1
            cur_sphere[np.where(cur_sphere[:, 2] >= np.shape(sphere_im)[2])[0], 2] = np.shape(sphere_im)[2] - 1
            
            sphere_im[cur_sphere[:, 0], cur_sphere[:, 1], cur_sphere[:, 2]] = sphere_im[cur_sphere[:, 0], cur_sphere[:, 1], cur_sphere[:, 2]] + 1
            
            if id_c % 1000000 == 0:
                print(str(id_c) + ' of total: ' + str(len(p)))
        
        
        ### Exclude regions with no tissue
        sphere_im[ref_atlas == 0] = 0
        
        
        
        ### also remove ventricles???
        
        
        ### scale to cells / mm^3  (currently is in cells/0.064 mm^3) ---> so multiply image by 
        
        cur_volume = ((radius * atlas_resolution[0] * 0.001) * 2)**3  ### in cells/mm^3
        scale_factor = 1/cur_volume    ### now scaled to cells/1 mm^3
        
        
        sphere_im = sphere_im * scale_factor
        
        
        ### save density image
        tiff.imwrite(sav_dir + exp['name'] + '_DENSITY_MAP_CERE_rad5_minsize30.tif', np.asarray(sphere_im, dtype=np.uint32))

        ### Plot single slice
        plt.figure(); plt.imshow(sphere_im[300], cmap='turbo')
        plt.colorbar(label='Density')     
        
        
 

### DEBUG: 
import napari
viewer = napari.Viewer()
viewer.add_image(sphere_im)
    





