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


from scipy import ndimage as ndi
from skimage import filters, segmentation, measure, morphology

import pickle

def remove_by_vol(im, thresh=200, binary=True):
    
    if not binary:  ### dont relabel if already a labelled array
        cc = measure.regionprops(im)        
    else:
        lab_im = measure.label(im)
        cc = measure.regionprops(lab_im)
    
    cleaned = np.zeros(np.shape(im))
    for id_c, obj in enumerate(cc):
        
        c = obj['coords']
        # val = obj['intensity_max']
        
        if len(c) > thresh:
            if not binary:
                cleaned[c[:, 0], c[:, 1], c[:, 2]] = id_c
            else:
                cleaned[c[:, 0], c[:, 1], c[:, 2]] = 1
            
    return cleaned


reference_atlas = '/home/user/.brainglobe/allen_mouse_20um_v1.2/annotation.tiff'


ref_atlas = tiff.imread(reference_atlas)
ref_atlas = np.asarray(ref_atlas, dtype=np.uint32)




        
#%% Load in control data
### ALSO have to load in average myelin map from ALL control brains to ensure can get matching myelin intensity density

list_brains = get_metadata(mouse_num = ['M296'])
input_path = list_brains[0]['path']
name_str = list_brains[0]['name']
n5_file = input_path + name_str + '.n5'
downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
atlas_dir_c1 = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
f = z5py.File(n5_file, "r")
dset_c1 = f['setup0/timepoint0/s0']  
  
### Make sure to also load in the shape of the downsampled raw data for later transforms!!!
dx_c1_shape = tiff.imread(atlas_dir_c1 + 'deformation_field_0_ADD_CERE.tiff').shape
        

# list_brains = get_metadata(mouse_num = ['M297'])
# input_path = list_brains[0]['path']
# name_str = list_brains[0]['name']
# n5_file = input_path + name_str + '.n5'
# downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
# atlas_dir_c2 = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
# f = z5py.File(n5_file, "r")
# dset_c2 = f['setup0/timepoint0/s0']  

# list_brains = get_metadata(mouse_num = ['M304'])
# input_path = list_brains[0]['path']
# name_str = list_brains[0]['name']
# n5_file = input_path + name_str + '.n5'
# downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
# atlas_dir_c3 = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
# f = z5py.File(n5_file, "r")
# dset_c3 = f['setup0/timepoint0/s0']  

# list_brains = get_metadata(mouse_num = ['M242'])       
# input_path = list_brains[0]['path']
# name_str = list_brains[0]['name']
# n5_file = input_path + name_str + '.n5'
# downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
# atlas_dir_c4 = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
# f = z5py.File(n5_file, "r")
# dset_c4 = f['setup0/timepoint0/s0']




#%% Compute inverted deformation fields

def compute_inverse_deformation_field_consistent(dx, dy, dz, atlas_shape, atlas_resolution_mm=0.02, tol_mm=0.1):
    """
    Compute an approximate inverse deformation field over the atlas grid.
    Maintains consistent dimension ordering (dim0, dim1, dim2) throughout.

    Parameters:
        dx, dy, dz: forward deformation fields (shape = source space), values in mm
                    dx = mapping to atlas dim 0, dy = to atlas dim 1, dz = to atlas dim 2
        atlas_shape: (dim0, dim1, dim2) tuple defining shape of the target atlas
        atlas_resolution_mm: spacing in mm (default 0.02 for 20 um)
        tol_mm: max KD-tree tolerance in mm

    Returns:
        inv_dx, inv_dy, inv_dz: inverse deformation fields (shape = atlas_shape),
                                values are voxel coordinates in source space
    """
    source_shape = dx.shape
    n_vox = np.prod(source_shape)

    # Stack into (N, 3) where each row = [d0, d1, d2] = atlas-space coords
    coords_atlas = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)

    # KD-tree on these forward-mapped points
    tree = cKDTree(coords_atlas)

    # Create a regular grid of atlas voxel centers in mm
    d0_idx, d1_idx, d2_idx = np.indices(atlas_shape)
    d0_mm = d0_idx * atlas_resolution_mm
    d1_mm = d1_idx * atlas_resolution_mm
    d2_mm = d2_idx * atlas_resolution_mm
    atlas_coords = np.stack([d0_mm.ravel(), d1_mm.ravel(), d2_mm.ravel()], axis=1)

    # Query inverse: which source voxel mapped closest to each atlas voxel?
    dists, source_indices = tree.query(atlas_coords, distance_upper_bound=tol_mm)
    valid = source_indices < n_vox

    inv_dx = np.full(atlas_coords.shape[0], np.nan, dtype=np.float32)
    inv_dy = np.full(atlas_coords.shape[0], np.nan, dtype=np.float32)
    inv_dz = np.full(atlas_coords.shape[0], np.nan, dtype=np.float32)

    # Convert flat source indices to source voxel coordinates
    src_d0, src_d1, src_d2 = np.unravel_index(source_indices[valid], source_shape)
    inv_dx[valid] = src_d0
    inv_dy[valid] = src_d1
    inv_dz[valid] = src_d2

    # Reshape back to atlas_shape
    inv_dx = inv_dx.reshape(atlas_shape)
    inv_dy = inv_dy.reshape(atlas_shape)
    inv_dz = inv_dz.reshape(atlas_shape)

    return inv_dx, inv_dy, inv_dz



def load_or_compute_inverse_field(atlas_shape, atlas_path, out_path, atlas_resolution_mm=0.02, tol_mm=0.1):
    """
    Checks for existing inverse field pickle. Computes and saves if not found.

    Parameters:
        dx, dy, dz: forward deformation fields in mm
        atlas_shape: shape of target atlas grid
        out_path: file path to .pkl output
        atlas_resolution_mm: resolution of atlas in mm
        tol_mm: KD-tree match tolerance in mm

    Returns:
        Dictionary with keys: 'inv_dx', 'inv_dy', 'inv_dz'
    """
    if os.path.exists(out_path):
        print(f"Loading inverse field from: {out_path}")
        with open(out_path, 'rb') as f:
            inverse_fields = pickle.load(f)
    else:
        print("Inverse field not found, computing...")
        
        deformation_field_paths = [atlas_dir_c1 + 'deformation_field_0_ADD_CERE.tiff',
                                   atlas_dir_c1 + 'deformation_field_1_ADD_CERE.tiff',
                                   atlas_dir_c1 + 'deformation_field_2_ADD_CERE.tiff'
                                  ]        

        dx = tiff.imread(deformation_field_paths[0])
        dy = tiff.imread(deformation_field_paths[1])
        dz = tiff.imread(deformation_field_paths[2])

        inv_dx, inv_dy, inv_dz = compute_inverse_deformation_field_consistent(
            dx, dy, dz, atlas_shape, atlas_resolution_mm, tol_mm
        )
        inverse_fields = {
            'inv_dx': inv_dx,
            'inv_dy': inv_dy,
            'inv_dz': inv_dz
        }
        print(f"Saving inverse field to: {out_path}")
        with open(out_path, 'wb') as f:
            pickle.dump(inverse_fields, f)

    return inverse_fields



#%% Computer inverted field for each control dataset and save (only if file doesn't exist)


inverse_path = atlas_dir_c1 + 'inverse_field.pkl'
inv_fields = load_or_compute_inverse_field(
    atlas_shape=ref_atlas.shape,
    atlas_path=atlas_dir_c1,
    out_path=inverse_path,
    atlas_resolution_mm=0.02,
    tol_mm=0.1
)

inv_dx = inv_fields['inv_dx']
inv_dy = inv_fields['inv_dy']
inv_dz = inv_fields['inv_dz']
  
 
    
 
    
 
    
#%% List of brains
# list_brains = get_metadata(mouse_num = ['M217'])

# list_brains = get_metadata(mouse_num = ['M243'])

# list_brains = get_metadata(mouse_num = 'all')


list_brains = get_metadata(mouse_num = [#'M243',
                                        # 'M244','M234', 
                                        'M235', 'M217']) ### 5xFAD
                                     #   'M304', 'M242', 'M297', 'M296'])        ### control





#%% Loop through each experiment
for exp in list_brains:

    input_path = exp['path']
    name_str = exp['name']
    
    n5_file = input_path + name_str + '.n5'
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'


    f = z5py.File(n5_file, "r")
    dset = f['setup0/timepoint0/s0']    
    dset_red = f['setup1/timepoint0/s0']    



    #%% Load everything necessary for atlas transforms later
    downsampled_dir = input_path + name_str + '_TIFFs_extracted/'
    atlas_dir = downsampled_dir + name_str + '_ANTS_registered/' + name_str  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'

    myelin_path = glob.glob(os.path.join(downsampled_dir,'*_ch0_n4_down1_resolution_20_PAD.tif'))[0] 
    myelin = tiff.memmap(myelin_path)


    reg_name = np.load(atlas_dir + 'brainreg_args.npy')
    z_after = int(reg_name[-1])
    x_after = int(reg_name[-2])
    
    # find actual down_factor while considering the padding on each side of the image
    m_shape = np.asarray(myelin.shape)
    m_shape[0] = m_shape[0] - z_after*2
    m_shape[1] = m_shape[1] - x_after*2
    m_shape[2] = m_shape[2] - x_after*2

    down_factor = dset.shape/m_shape
    
    ### Even load in deformation fields
    deformation_field_paths = [atlas_dir + 'deformation_field_0_ADD_CERE.tiff',
                               atlas_dir + 'deformation_field_1_ADD_CERE.tiff',
                               atlas_dir + 'deformation_field_2_ADD_CERE.tiff'
                              ]
    dx = tiff.imread(deformation_field_paths[0])
    dy = tiff.imread(deformation_field_paths[1])
    dz = tiff.imread(deformation_field_paths[2])
    deformation_fields = [dx, dy, dz]





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
    sav_dir = input_path + '/' + filename + '_postprocess_CONGO'
    
    congo_dir = input_path + name_str + '_MaskRCNN_patches_CONGO/'
    
    
    
    import re
    """ For testing ILASTIK images """
    images_congo = glob.glob(os.path.join(congo_dir,'*_df.pkl'))    # can switch this to "*truth.tif" if there is no name for "input"
    images_congo.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # images_congo = [int(re.search(r'(\d+)_input_im\.tif$', path).group(1)) for path in images]
    congo_examples = [dict(pkl=i, segmentation=i.replace('_df.pkl','_segmentation.tif'),
                     input=i.replace('_df.pkl', '_input_im.tif')) for i in images_congo]

    halo_dir = input_path + name_str + '_MaskRCNN_patches_HALO/'
    images_halo = glob.glob(os.path.join(halo_dir,'*_df.pkl'))    # can switch this to "*truth.tif" if there is no name for "input"
    images_halo.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # integers_halo = [int(re.search(r'(\d+)_input_im\.tif$', path).group(1)) for path in images]
    halo_examples = [dict(pkl=i, segmentation=i.replace('_df.pkl','_segmentation.tif'),
                     input=i.replace('_df.pkl', '_red_im.tif')) for i in images_halo]
     
    try:
        # Create target Directory
        os.mkdir(sav_dir)
    except FileExistsError:
        print("\nSave directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    
    

    if len(congo_examples) - len(halo_examples)  != 0:
        print('MISSING PICKLES')
        zzz
        
        
        
    # # Function to extract the integer ID
    # def extract_id(path, string='input_im'):
    #     match = re.search(r'(\d+)_' + string + '\.tif$', path)
    #     return int(match.group(1)) if match else None
    
    # # Create mappings: id -> index
    # A_ids = [extract_id(path, string='input_im') for path in images_congo]
    # B_ids = [extract_id(path, string='red_im') for path in images_halo]
    
    # # Create dictionaries to map id to index
    # A_id_to_index = {id_: idx for idx, id_ in enumerate(A_ids) if id_ is not None}
    # B_id_to_index = {id_: idx for idx, id_ in enumerate(B_ids) if id_ is not None}
    
    # # Find shared IDs
    # shared_ids = set(A_id_to_index.keys()) & set(B_id_to_index.keys())
    
    # # Get the matching indexes
    # A_matching_indexes = [A_id_to_index[i] for i in shared_ids]
    # B_matching_indexes = [B_id_to_index[i] for i in shared_ids]
    
    # # (Optional) Sort indexes to maintain original order
    # A_matching_indexes.sort()
    # B_matching_indexes.sort()
    
    # # Get shortened lists based on those indexes
    # congo_examples = [congo_examples[i] for i in A_matching_indexes]
    # halo_examples = [halo_examples[i] for i in B_matching_indexes]
    



    for blk_num in range(0, len(congo_examples)):   ### 288 is the good one to test?

    
                                                    ### 370 is alveus   3## 369 pretty good though too
        pkl_path = sav_dir + filename + '_HALO_PARSED_' + str(int(blk_num)) + '_df.pkl'
        if os.path.exists(pkl_path):
            print("Pickle found — skipping generation.")
            continue  
            
         
        ### Read in pickle for halos
        pkl = pd.read_pickle(congo_examples[blk_num]['pkl'])




        if len(pkl) == 0:
            print('SKIP, no pickle, no objects')
            continue
        
        
        s_c = pkl['xyz_offset'][0]
        
        
        
        ### Load halo segmentations and red_im (input_im)
        halo_seg = tiff.imread(halo_examples[blk_num]['segmentation'])
        input_im = tiff.imread(halo_examples[blk_num]['input'])
                
        
        ### Then make the congo seg
        # pkl_congo = pd.read_pickle(congo_examples[blk_num]['pkl'])
        
        congo_seg = np.zeros(np.shape(halo_seg))
        
        for id_core, core in pkl.iterrows():
            
            c = core['coords_raw']
            congo_seg[c[:, 0], c[:, 1], c[:, 2]] = 1
            
        
        
        # congo_seg = tiff.imread(congo_examples[blk_num]['segmentation'])
        
        
        
        congo_seg = remove_by_vol(congo_seg, thresh=50, binary=True)  ### remove very small debris
        

        ### Some areas are ZERO background (due to stitching weirdness) --- add to training data later
        ### For now, just make out with input_im
        # plot_max(halo_seg)
        congo_seg[input_im == 0] = 0
        halo_seg[input_im == 0] = 0
        # plot_max(halo_seg)
        
        
        

        ### remove small cores (slightly more strict because don't want tiny cores in halo consideration)
        congo_clean = remove_by_vol(congo_seg, thresh=200, binary=True)
        
            
        ### REMOVE ANY UNMATCHED MARKERS
        congo_clean[halo_seg == 0] = 0
        ### Run watershed
        markers_labeled, _ = ndi.label(congo_clean)  # Give each sphere a unique label
        distance = ndi.distance_transform_edt(halo_seg)
        inverted_distance = -distance
        labels = segmentation.watershed(inverted_distance, markers_labeled, mask=halo_seg)
        
        ### clean up fragments left by watershed
        labels = remove_by_vol(labels, thresh=300, binary=False)



        ### Add back in whatever halos were not matched
        # first find which ones were not matched
        unmatched_halos = np.copy(halo_seg)
        unmatched_halos[labels > 0] = 0
    
        
        # Now threshold by size --- make it pretty severe, only want large halos
        cleaned = remove_by_vol(unmatched_halos, thresh=1000, binary=True)
        
        
        # And then add these into the labels array with unique IDs
        cleaned_lab, _ = ndi.label(cleaned)
        cleaned_lab = cleaned_lab + np.max(labels) + 1
        cleaned_lab[cleaned_lab == np.max(labels) + 1] = 0 # reset background
        
        labels = labels + cleaned_lab
        
        
        
        
        ### Now, get a similarily colored labelled core array, which can be referenced later to figure out associations
        cores = np.copy(congo_seg)
        cores[labels == 0] = 0
        
        colored_cores = np.copy(labels)
        colored_cores[cores == 0] = 0
        # colored_cores = np.asarray(colored_cores, dtype=np.uint64)
        
    
        ### also get unmatched cores which will NOT be counted as main cores later --- use congo seg, so NO size filter
        unmatched_cores = np.copy(congo_seg)  
        unmatched_cores[colored_cores > 0] = 0
        
        ###^^^ not quite accurate, because halos may have sliced cores into pieces, we want instead to add back in any unmatched_cores
        ###    that are already touching completed colored_cores... have to do this at the "colored_cores" step
    
        
        ### --- optional --- maybe can add in also small sphere for any congo core that has no or small halo?
        
        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(halo_seg)
        # viewer.add_image(congo_seg)
        # viewer.add_image(congo_clean)
        # viewer.add_image(labels)
            
        
        # viewer.add_image(input_im)
        # # viewer.add_image(labels)
        # viewer.add_image(colored_cores)
        # viewer.add_image(unmatched_cores)
        
        # viewer.show(block=True)
        
        # tiff.imwrite(sav_dir + filename + '_blk_num_' + str(blk_num) + '_WATERSHED_input_im', np.asarray(input_im, dtype=np.uint16))
        # tiff.imwrite(sav_dir + filename + '_blk_num_' + str(blk_num) + '_WATERSHED_labels', np.asarray(labels, dtype=np.uint16))
        # tiff.imwrite(sav_dir + filename + '_blk_num_' + str(blk_num) + '_WATERSHED_colored_cores', np.asarray(colored_cores, dtype=np.uint16))

        # myelin_im = dset[s_c[2]:s_c[2]+input_im.shape[0],s_c[1]:s_c[1]+input_im.shape[1],s_c[0]:s_c[0]+input_im.shape[2]]
        # tiff.imwrite(sav_dir + filename + '_blk_num_' + str(blk_num) + '_WATERSHED_myelin_im.tif', np.asarray(myelin_im, dtype=np.uint16))


        # congo_im = tiff.imread(congo_examples[blk_num]['input'])
        # tiff.imwrite(sav_dir + filename + '_blk_num_' + str(blk_num) + '_WATERSHED_congo_im.tif', np.asarray(congo_im, dtype=np.uint16))



        #%% 
        ### Then must save as pickle dataframe --- will also need to get intensity value for each object from myelin image!!!
        
        # Dataframe per row is --- core centroid, core volume, core coords(?), core XY area(?) core intensity, core myelin intensity density
        #                      --- halo volume, halo XY area(?), halo coords(?), halo intensity, halo myelin intensity density


        # cc_halos = measure.regionprops(labels)

        print('Running faster regionprops')
        
        # import numpy as np
        from scipy.ndimage import center_of_mass, find_objects, mean
        from collections import defaultdict

        # Inputs
        labels = measure.label(labels)  ### reset to make sure values are consecutive
        
        
        ### Also get associated colored cores
        colored_cores = np.copy(labels)
        colored_cores[cores == 0] = 0
        
        
        
        label_ids = np.unique(labels)
        label_ids = label_ids[label_ids != 0]  # exclude background
        
        # Properties
        centroids = center_of_mass(np.ones_like(labels), labels=labels, index=label_ids)
        volumes   = np.bincount(labels.ravel())[label_ids]
        
        # try:
        #     core_vols = np.bincount(colored_cores.ravel())[label_ids]
        # except:
        #     print('No core vols')
        #     core_vols = []
            
        ### --- Robust core volume calculation ---
        core_counts = np.bincount(colored_cores.ravel())
        core_vols_all = np.zeros(labels.max() + 1, dtype=int)  # initialize full-sized array that is size of LABELS and not core labels, so won't run into error later!!!
        core_labels = np.unique(colored_cores)
        core_labels = core_labels[core_labels != 0]  ### remove 0 ID
        core_vols_all[core_labels] = core_counts[core_labels]
        core_vols = core_vols_all[label_ids]  # extract only those that match label_ids
        ### --------------------------------------
        
           
        bboxes    = find_objects(labels)
        # intensities = mean(intensity_image, labels=labels, index=label_ids)

        ### Get coordinates of halos
        coords = np.array(np.nonzero(labels)).T
        labels_flat = labels[tuple(coords.T)]
        
        coord_dict = defaultdict(list) ### SPEEDS UP --- instead of having to search through all coords later
        for coord, label in zip(coords, labels_flat):
            coord_dict[label].append(coord)
            
        # Convert to arrays
        for label in coord_dict:
            coord_dict[label] = np.array(coord_dict[label])           
        
            
        ### Also get coords of core
        coords_core = np.array(np.nonzero(colored_cores)).T
        labels_flat = labels[tuple(coords_core.T)]
        
        coord_dict_cores = defaultdict(list) ### SPEEDS UP --- instead of having to search through all coords later
        for coord, label in zip(coords_core, labels_flat):
            coord_dict_cores[label].append(coord)
            
        # Convert to arrays
        for label in coord_dict_cores:
            coord_dict_cores[label] = np.array(coord_dict_cores[label])            
        
        
        
        
        data = []
        for i, label in enumerate(label_ids):
            # print(i)
            prop = {
                'centroid': centroids[i],
                'volume': volumes[i],
                'core_vols': core_vols[i],  # fixed: per-label core volume
            }
            if label < len(bboxes) and bboxes[label - 1] is not None:
                
                # zzz
                bbox_slices = bboxes[label - 1]
                x0, x1 = bbox_slices[2].start, bbox_slices[2].stop
                y0, y1 = bbox_slices[1].start, bbox_slices[1].stop
                z0, z1 = bbox_slices[0].start, bbox_slices[0].stop
                prop['bbox'] = (z0, y0, x0, z1, y1, x1)
            else:
                prop['bbox'] = None  # No bounding box


            prop['coords_halo'] = coord_dict[label]
            prop['coords_core'] = coord_dict_cores[label]
    
            data.append(prop)
    
        df = pd.DataFrame(data)

        df['xyz_offset'] = [s_c] * len(df)
        df['block_num'] = blk_num
        
        ### Maybe no bounding box (if object is on edge???) - so drop
        df = df.dropna()


        #%% Loop through by halo
        
        from tqdm import tqdm
        
        d = {}
        for i_list, halo in tqdm(df.iterrows(), total=len(df), desc="Processing halos"):
            # halo = df.iloc[281]
            
            # print(i_list)
            bbox = np.copy(halo['bbox'])
            coords_halo = np.copy(halo['coords_halo'])
            coords_scaled = np.copy(coords_halo)
            
            
            coords_halo[:, 0] = coords_halo[:, 0] - bbox[0]
            coords_halo[:, 1] = coords_halo[:, 1] - bbox[1]
            coords_halo[:, 2] = coords_halo[:, 2] - bbox[2]
            
            
            bbox[0] = bbox[0] + s_c[2]  ### z
            bbox[3] = bbox[3] + s_c[2]
            
            bbox[1] = bbox[1] + s_c[1]  ### x
            bbox[4] = bbox[4] + s_c[1]
            
            bbox[2] = bbox[2] + s_c[0]  ### y
            bbox[5] = bbox[5] + s_c[0]
        
            ### GET INTENSITY OF HALO from current im, and also all controls
            myelin_crop = dset[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]

            # also get coordinates of halo within the crop to do mask out and find myelin intensity
            myelin_intensity = np.mean(myelin_crop[coords_halo[:, 0], coords_halo[:, 1], coords_halo[:, 2]])
            
            # plot_max(myelin_crop)
            # red_crop = dset_red[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
            # halo_crop = np.zeros(np.shape(myelin_crop))
            # halo_crop[coords_halo[:, 0], coords_halo[:, 1], coords_halo[:, 2]] = 1
            # plot_max(red_crop)
            # plot_max(halo_crop)

            
            # import napari
            # viewer = napari.Viewer()
            # viewer.add_image(myelin_crop)
            # viewer.add_image(red_crop)
            # viewer.add_image(halo_crop)
            # viewer.show(block=True)
        
        
            ### Also get scaled coords?            
            coords_scaled[:, 0] = coords_scaled[:, 0] + s_c[2]
            coords_scaled[:, 1] = coords_scaled[:, 1] + s_c[1]
            coords_scaled[:, 2] = coords_scaled[:, 2] + s_c[0]
            

            d[i_list] = {
                    'bbox_dset':bbox,
                    'coords_scaled':coords_scaled,
                    'myelin_int':myelin_intensity,
                    }
         


        extra_df = pd.DataFrame.from_dict(d, "index")

        df = pd.concat([df, extra_df], axis=1)

        
        if len(df) == 0:
            print('No halos')
            continue  
        
        #%% 
        print('Transforming bounding boxes to atlas')


        # Function to extract and scale the top-left corner
        def scale_top_left(bbox):
            
            # Scaling factors
            # scale_x = XY_res / 20
            # scale_y = XY_res / 20
            # scale_z = Z_res / 20
        
            z0, x0, y0 = bbox[:3]
            return np.round([int(z0 / down_factor[0] + z_after), int(x0 / down_factor[1] + x_after), int(y0 / down_factor[2] + x_after)])
        
        # Apply and add as new column
        df['bbox_top_left_20um'] = df['bbox_dset'].apply(scale_top_left)
                
                


        #%%
        ### If want to debug with s_c, must swap 0 and 2 positions [1280, 5120, 1024] ---> [1024, 5120, 1280]
        ### For debugging, set 
        # points = np.vstack([s_c, s_c])
        # points[:, [2, 1, 0]] = points[:, [0, 1, 2]]   # then rotate columns 
        # # and run the scale_top_left() function
        # points = scale_top_left(points[0])
        # points = np.vstack([points, points])
        
        
        
        ### Convert coordinates into ATLAS reference frame
        points = np.vstack(df['bbox_top_left_20um'])
        # and then move columns (1 to 0)  --> all these steps just do the inverse of original steps
        points[:, [1, 0, 2]] = points[:, [0, 1, 2]]
        


        ### Flip Z axis
        # downsampled_points[:, 0] = deformation_field.shape[0] - downsampled_points[:, 0]
        points[:, 1] = dx.shape[1] - points[:, 1]
        points[:, 2] = dx.shape[2] - points[:, 2]
                

        atlas_resolution = [20, 20, 20]

        field_scales = [int(1000 / resolution) for resolution in atlas_resolution]
        mapped_points = [[], [], []]
        for axis, deformation_field in enumerate(deformation_fields):
            # deformation_field = tiff.imread(deformation_field_path)
            print('hello')
            for point in points:
                point = [int(round(p)) for p in point]
                
            
                mapped_points[axis].append(
                    ### REMOVED ROUNDING - TIGER --- for Congo analysis
                    #int(
                    #    round(
                            field_scales[axis] * deformation_field[point[0], point[1], point[2]]
                    #    )
                    #)
                )
                
        mapped_points = np.transpose(mapped_points)
        
        p = np.copy(mapped_points)
        


        #%% Now map onto control brains!!! --- do everything backwards               
        p = np.round(p).astype(int)
        
        
        # Clip coordinates to valid range for each axis
        for dim in range(3):
            p[:, dim] = np.clip(p[:, dim], 0, inv_dx.shape[dim] - 1)
    

        rx = inv_dx[p[:, 0], p[:, 1], p[:, 2]]
        ry = inv_dy[p[:, 0], p[:, 1], p[:, 2]]
        rz = inv_dz[p[:, 0], p[:, 1], p[:, 2]]


        rp = np.transpose(np.vstack([rx, ry, rz]))


        print('Applying remapped points to control brains')




        #%% Now have to scale these remapped points back to the high resolution of the original dataset!!!
        
        ### First flip along Z axis
        # downsampled_points[:, 0] = deformation_field.shape[0] - downsampled_points[:, 0]
        rp[:, 1] = dx_c1_shape[1] - rp[:, 1]
        rp[:, 2] = dx_c1_shape[2] - rp[:, 2]
        
        ### Convert coordinates into DSET reference frame        
        rp[:, [0,1,2]] = rp[:, [1,0,2]]


        df['bbox_top_c1'] = [np.asarray(voxel).astype(int) if not np.any(np.isnan(voxel)) else None for voxel in rp]
        
        

        # Function to extract and scale the top-left corner
        def scale_up_left(bbox):
            
            if bbox is None:
                return None
            z0, x0, y0 = bbox[:3]
            return np.round([int((z0 - z_after) * down_factor[0]), int((x0 - x_after) * down_factor[1]), int((y0 - x_after) * down_factor[2])])
        
        # Apply and add as new column
        df['bbox_top_c1'] = df['bbox_top_c1'].apply(scale_up_left)
                


        #%% Now can go to control data and find fluorescence using these bounding box tips 
        ###         --- need to re-run mapping for EACH control brain though :O
        

        def compute_new_bbox(row):
            x0, y0, z0, x1, y1, z1 = row['bbox_dset']

            if row['bbox_top_c1'] is None:
                return None   
            a0, b0, c0 = row['bbox_top_c1']
            
        
            
            
            # Compute size of original bbox
            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            
            # Apply size to new corner
            a1 = a0 + dx
            b1 = b0 + dy
            c1 = c0 + dz
            
            return [a0, b0, c0, a1, b1, c1]
        
        # Apply to DataFrame
        df['bbox_dset_c1'] = df.apply(compute_new_bbox, axis=1)

        df = df.dropna()



        print('Extracting myelin intensity from control brain')
        myelin_arr = []
        for i_r, halo in df.iterrows():
            
            # zzz
            # print(i_r)
            bbox = halo['bbox_dset_c1']
            
            ### SKIP IF NEGATIVE --- MEANS OUT OF BOUNDS
            if any(x < 0 for x in bbox):
                myelin_arr.append(-1)
                continue
            
            bbox_raw = halo['bbox_dset']
            
            coords_scaled = halo['coords_scaled']
            # coords_scaled = np.roll(coords_scaled, 1)  ### NOT NEEDED LATER
            coords_crop = np.copy(coords_scaled)

            
            coords_crop[:, 0] = coords_scaled[:, 0] - bbox_raw[0]
            coords_crop[:, 1] = coords_scaled[:, 1] - bbox_raw[1]
            coords_crop[:, 2] = coords_scaled[:, 2] - bbox_raw[2]
            

                        
            def is_bbox_in_bounds(bbox, volume_shape):
                z0, y0, x0, z1, y1, x1 = bbox
                Z, Y, X = volume_shape
            
                return (
                    0 <= z0 < z1 <= Z and
                    0 <= y0 < y1 <= Y and
                    0 <= x0 < x1 <= X
                )
            
            # Example usage
            if is_bbox_in_bounds(bbox, dset_c1.shape):
                myelin_crop_c1 = dset_c1[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
    
                myelin_intensity = np.mean(myelin_crop_c1[coords_crop[:, 0], coords_crop[:, 1], coords_crop[:, 2]])
                        
                myelin_arr.append(myelin_intensity)
                
                
            else:
                print("Skipping: bbox is out of bounds")
                myelin_arr.append(np.nan)
                
    
           
            
            
            # plot_max(myelin_crop_c1)
            # halo_crop_c1 = np.zeros(np.shape(myelin_crop))
            # halo_crop_c1[coords_crop[:, 0], coords_crop[:, 1], coords_crop[:, 2]] = 1
            # plot_max(halo_crop_c1)

            
            # import napari
            # viewer = napari.Viewer()
            # viewer.add_image(myelin_crop)
            # viewer.add_image(red_crop)
            # viewer.add_image(halo_crop)
            
            # viewer.add_image(myelin_crop_c1)
            # viewer.add_image(halo_crop_c1)
                        
            # viewer.show(block=True)
            
            
            # zzz
            
            
            #%%
            # ### For debug --- find closest to top left corner so can check if the overall FOV is similar or not
            # coords_array = np.stack(df['bbox_top_c1'].values)
            # top_corner = np.min(coords_array, axis=0)  ### minimum should be close enough to top left corner
            
            # top_corner = [1079, 5084, 1526]  ### top left corner after transforms --- might want to check bottom right corner as well?
            
            
            # # For alveus testing
            # top_r = [ 1199, 6334, 2758]
  
            
            
            # Lpatch_size = 128 * 10
            # Lpatch_depth = 64 * 4
            # # patch = dset_c1[top_corner[0]:top_corner[0] + Lpatch_depth, top_corner[1]:top_corner[1] + Lpatch_size, top_corner[2]:top_corner[2] + Lpatch_size]

            # patch1 = dset_c1[top_r[0]:top_r[0] + Lpatch_depth, top_r[1]:top_r[1] + Lpatch_size, top_r[2]:top_r[2] + Lpatch_size]


            # im_patch = dset[s_c[2]:s_c[2] + Lpatch_depth, s_c[1]:s_c[1] + Lpatch_size, s_c[0]:s_c[0] + Lpatch_size]
                        
            
            # halo_im = tiff.imread(halo_examples[blk_num]['input'])
            
            
            # import napari
            # viewer = napari.Viewer()
            # viewer.add_image(im_patch)
            # viewer.add_image(patch1)
            # viewer.add_image(halo_im)
            # viewer.show(block=True)
                        
            
        myelin_arr = np.vstack(myelin_arr)
        df['c1_myelin'] = myelin_arr




        ### REMOVE OL CELL BODIES???

        df.to_pickle(sav_dir + filename + '_HALO_PARSED_' + str(int(blk_num)) + '_df.pkl')






        ### Make average summed morphology of what FORNIX vs. CORTEX congo plaques look like!!!





        
          
    
