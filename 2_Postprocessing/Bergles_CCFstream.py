#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:10:46 2023

@author: user
"""

import nrrd
import numpy as np
import tifffile as tiff

from postprocess_func import *

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ccf_streamlines.projection as ccfproj
import skimage

import sys
sys.path.append("..")

from get_brain_metadata import *




""" Files to download:
3D CCF-aligned data to two-dimensional views. As an example, we will use the
`10-micron resolution average template data <http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/average_template/average_template_10.nrrd>`_ for the CCF.

      * - `top.h5 <https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/cortical_coordinates/ccf_2017/ccf_streamlines_assets/view_lookup/top.h5>`_
- Correspondence between view from top of isocortex and CCF volume. Can place hemispheres adjacent to each other with ``view_space_for_other_hemisphere = True``.
- .. image:: /images/top_template.png
     :width: 400
     
     * - `surface_paths_10_v3.h5 <https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/cortical_coordinates/ccf_2017/ccf_streamlines_assets/streamlines/surface_paths_10_v3.h5>`_
- Contains the (linear) voxel locations of each streamline and a lookup table to find a streamline for each voxel. **Warning:** Large file (~0.5 GB)


    * - `flatmap_butterfly.h5 <https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/cortical_coordinates/ccf_2017/ccf_streamlines_assets/view_lookup/flatmap_butterfly.h5>`_
      - Correspondence between flattened map of all of isocortex and CCF volume - medial side has been adjusted so that the hemispheres are abutting at the center. Can place hemispheres adjacent to each other with ``view_space_for_other_hemisphere = 'flatmap_butterfly'``.
"""

### this is the same atlas and orientation as the one from Allen Atlas from BrainReg in /home/user/.brainglobe
# our Perens and Princeton atlases are 20um not 10um
# template, _ = nrrd.read("./extra_downloads/average_template_10.nrrd")
#input_path = '/media/user/c0781205-1cf9-4ece-b3d5-96dd0fbf4a78/20231012_M230_MoE_PVCre_SHIELD_delip_RIMS_RI_1500_3days_5x/M230_fused_TIFFs_extracted/M230_fused_ISOCORTEX_CORTEX_ONLY_allen_mouse_10um_bend_0.95_grid_-10_gauss_0/'
#input_path = '/media/user/ce86e0dd-459a-4cf1-8194-d1c89b7ef7f6/20231031_M223_MoE_PVCre_P56_SHIELD_delp_RIMS_50perc_then_100perc_expanded_slightly_more_5x/M223_fused_TIFFs_extracted/M223_fused_ISOCORTEX_allen_mouse_10um_bend_0.95_grid_-10_gauss_0/'
# input_path = '/media/user/c0781205-1cf9-4ece-b3d5-96dd0fbf4a78/20231115_M124_MoE_CasprtdT_Cuprizone_6wk__SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER/M124_cuprizone_fused_TIFFs_extracted/M124_cuprizone_fused_ISOCORTEX_CORTEX_ONLY_allen_mouse_10um_bend_0.95_grid_-10_gauss_0/'
# input_path = '/media/user/20TB_HDD_NAS1/20240210_M254_MoE_P60_low_AAVs_SHIELD_CUBIC_7d_RIMS_2d_RI_1493_sunflower/M254_MoE_P60_low_AAVs_fused_TIFFs_extracted/M254_MoE_P60_low_AAVs_fused_ISOCORTEX_CORTEX_ONLY_allen_mouse_10um_bend_0.95_grid_-15_gauss_0/'
# input_path = '/media/user/8TB_HDD/20240216_M246_96rik_with_nanobodies_SHIELD_CUBIC_RIMS_RI_1493_sunflow/M246_96rik_nanobodies_P60_fused_TIFFs_extracted/M246_96rik_nanobodies_P60_fused_ISOCORTEX_CORTEX_ONLY_allen_mouse_10um_bend_0.95_grid_-15_gauss_0/'





CUBIC = 0
list_brains = get_metadata(mouse_num = ['M254'])   ### P60
# list_brains = get_metadata(mouse_num = ['M286'])    # P240 brain
# list_brains = get_metadata(mouse_num = ['M271'])    # P620 brain
# list_brains = get_metadata(mouse_num = ['Otx6'])    # P950/800 brain


# CUBIC = 1
# list_brains = get_metadata(mouse_num = ['M265'])   ### Cuprizone
# list_brains = get_metadata(mouse_num = ['M310'])   ### Recovery



#%% Select atlas directory --- containing registered brains

### NEED TO DO OTHER 10 um registrations!!!
# input_path = list_brains[0]['path'] + list_brains[0]['name'] + '_TIFFs_extracted/' + list_brains[0]['name'] + 'allen_mouse_10um_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_0.9_n4_1_grid_-15_gauss_0_use_steps_default_PADDED_50/'

### USE 20 um and upscale
if not CUBIC:
    input_path = list_brains[0]['path'] + list_brains[0]['name'] + '_TIFFs_extracted/' + list_brains[0]['name'] + '_ANTS_registered/' + list_brains[0]['name']  + 'allen_mouse_MYELIN_20um_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'
else:
    # for CUBIC brains
    input_path = list_brains[0]['path'] + list_brains[0]['name'] + '_TIFFs_extracted/' + list_brains[0]['name'] + '_ANTS_registered/' + list_brains[0]['name']  + 'allen_mouse_20um_CUBIC_CORTEX_ONLY_DIVIDE_MYELIN_0.9_n4_1_grid_-10_gauss_0_use_steps_default/'



upscale = 1


filename = list_brains[0]['name'] 
          
sav_dir = input_path
#im = tiff.imread(input_path + 'downsampled_standard_downsampled_standard.tiff')

#im = tiff.imread(input_path + 'downsampled_standard_downsampled_standard_M230_fused_level_s3_ch0_PAD.tiff')
#im = tiff.imread(input_path + 'downsampled_standard_downsampled_standard_M223_fused_level_s3_ch0_PAD.tiff')

# im = tiff.imread(input_path + 'downsampled_standard_downsampled_standard_M254_MoE_P60_low_AAVs_fused_level_s3_ch0_n4_down1_PAD.tiff')

# proj_type = 'raw'  ### OR
# proj_type = 'density'
proj_type = 'LARGE_density'

kind = 'max'
# kind = 'mean'

if proj_type == 'raw':
    
    if not upscale:
        im = tiff.imread(input_path + 'downsampled_standard_' + list_brains[0]['name'] + '_level_s3_ch0_n4_down1_resolution_10_PAD.tiff')
    
    else:
        if not CUBIC:
            im = tiff.imread(input_path + 'downsampled_standard.tiff')
        else:
            im = tiff.imread(input_path + 'downsampled_standard_'  + list_brains[0]['name'] + '_ANTS_myelin_cortex_only.tiff')
        print('scaling up RAW data')
        im = rescale(im, 2, preserve_range=True)
    
    
        

elif proj_type == 'density':
    ### LOAD IN DENSITY MAP AND SCALE IT UP!
    # im = tiff.imread(list_brains[0]['path'] + list_brains[0]['name'] + '_postprocess/' + list_brains[0]['name'] + '_DENSITY_MAP.tif')
    # im = tiff.imread('/media/user/20TB_HDD_NAS/20240103_M126_MoE_Reuse_delip_RIMS_RI_14926_sunflow_80perc/M126_MoE_P60_fused_postprocess/M126_MoE_P60_fused_DENSITY_MAP_CERE.tif')
    
    # sav_dir = list_brains[0]['path'] + list_brains[0]['name'] + '_postprocess/'
    # filename = 'P60_rad5'
    # filename = 'P240_rad5'
    # filename = 'P620_rad5'
    # filename = 'P800_rad5'
    # filename = 'CUPRIZONE_rad5'
    # filename = 'RECOVERY_rad5'
    # filename = 'FVB_rad5'
    filename = 'CD1_rad5'
    
    sav_dir = '/media/user/8TB_HDD/Mean_autofluor/'
    
    ### Load up density map
    im = tiff.imread(sav_dir + filename + '_mean_density_MAP.tif')
    
    print('scaling up density map')
    im = rescale(im, 2, preserve_range=True)


elif proj_type == 'LARGE_density':    
    ### LOAD IN DENSITY MAP AND SCALE IT UP!
    # filename = 'P60_LARGE'
    # filename = 'P240_LARGE'
    # filename = 'P620_LARGE'
    # filename = 'P850_LARGE'
    # filename = 'CUPRIZONE_LARGE'
    filename = 'RECOVERY_LARGE'
    # filename = 'FVB_LARGE'
    # filename = 'CD1_LARGE'
    
    sav_dir = '/media/user/8TB_HDD/Mean_autofluor/'
    
    ### Load up density map
    im = tiff.imread(sav_dir + filename + '_mean_density_MAP.tif')   
    
    
    sav_dir = sav_dir + 'LARGE_flatmounts/'
    
    print('scaling up density map')
    im = rescale(im, 2, preserve_range=True)

    
    



#%% mask out everything except layer that we want!
atlas = tiff.imread('/home/user/.brainglobe/allen_mouse_10um_v1.2/annotation.tiff')


with open('../atlas_ids/atlas_ids.json') as json_file:
    data = json.load(json_file)
 
data = data['msg'][0]         

    
keys_dict = get_ids_all(data, all_keys=[], keywords=[''])  
keys_df = pd.DataFrame.from_dict(keys_dict)

cc_allen = regionprops(atlas, cache=False)
cc_labs_allen = [region['label'] for region in cc_allen]
cc_labs_allen = np.asarray(cc_labs_allen)


#%% Get coordinates of cells only in gray matter to plot 
sub_idx = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Isocortex')
sub_keys = keys_df.iloc[sub_idx]
sub_keys.reset_index(inplace=True, drop=True)

layers = ['1', '2/3', '5', '6']

for layer in layers:
    
    
    ### Set different vmax for each layer
    if layer == '1':
        print('what')
        vmax = 6000
    elif layer == '2/3' or layer == '5':
        vmax = 20000
    elif layer == '6':
        vmax = 35000
        
        
    if proj_type == 'LARGE_density':
        if layer == '1':
            print('what')
            vmax = 450
            
            if filename == 'RECOVERY_LARGE':
                vmax = 2500
            
        elif layer == '2/3' or layer == '5':
            vmax = 450
            if filename == 'RECOVERY_LARGE':
                vmax = 2500

        elif layer == '6':
            vmax = 450
            if filename == 'RECOVERY_LARGE':
                vmax = 2500
            

    
    
    
    
    print('projecting layer: ' + layer)
    ### get layer 1 ids
    sub_rows = get_ids_of_layer_from_keys(sub_keys, layer=layer)    
    
    ### APPEND LAYER 4 if layer 2/3
    if layer == '2/3':
        sub_4 = get_ids_of_layer_from_keys(sub_keys, layer='4')
        sub_rows = sub_rows + sub_4  ### concatenate list
        
        
   
    
    sub_ids = np.asarray(sub_keys['ids'][sub_rows])
    
    """ Mask out just the layer using Allen atlas """
    ### do for Allen atlas
    iso_layer = np.zeros(np.shape(atlas))
    for idx in sub_ids:
        cur_id = np.where(cc_labs_allen == idx)[0]
        
        #print(cur_id)
        if len(cur_id) == 0:  ### if it does not exists in atlas
            continue
        cur_coords = cc_allen[cur_id[0]]['coords']
        iso_layer[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = idx
       
    
    ### mask out myelin
    template = np.copy(im)
    template[iso_layer == 0] = 0
    


    #%% Project onto cortex
    
    """
    Now, we can set up our projector object to perform operations on this 3D data.
    We'll first need a few other files, which you can find listed in :ref:`Data Files`.
    For this example, we'll be using the ``top.h5`` view lookup file and the
    ``surface_paths_10_v3.h5`` streamlines file. These will be given to a
    :class:`~ccf_streamlines.projection.Isocortex2dProjector` object that will
    process the 3D data.
    """
    
    proj_top = ccfproj.Isocortex2dProjector(
        # Specify our view lookup file
        projection_file="./extra_downloads/top.h5",
    
        # Specify our streamline file
        surface_paths_file="./extra_downloads/surface_paths_10_v3.h5",
    
        # Specify that we want to project both hemispheres
        hemisphere="both",
    
        # The top view contains space for the right hemisphere, but is empty.
        # Therefore, we tell the projector to put both hemispheres side-by-side
        view_space_for_other_hemisphere=True,
    )
    
    
    """
    We can now project the volume to a 2D top view. The default way of consolidating
    the information along the streamline is to use a maximum intensity projection
    (so the highest value along the streamline is used in the 2D image).
    """
    
    
    if proj_type=='raw':
        top_projection_max = proj_top.project_volume(template)
        cmap='Greys_r'
        
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w=8,h=8)
        ax = plt.Axes(fig, [0., 0., 0.9, 0.9])
        ax.set_axis_off()
        fig.add_axes(ax)
    
        
        a = ax.imshow(
            top_projection_max.T, # transpose so that the rostral/caudal direction is up/down
            interpolation='none',
            cmap=cmap,
        )       
        
        
    elif proj_type=='density'  or proj_type=='LARGE_density':
        top_projection_max = proj_top.project_volume(template, kind=kind)
        cmap = 'turbo'
    
    
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w=8,h=8)
        ax = plt.Axes(fig, [0., 0., 0.9, 0.9])
        ax.set_axis_off()
        fig.add_axes(ax)
    
        
        a = ax.imshow(
            top_projection_max.T, # transpose so that the rostral/caudal direction is up/down
            interpolation='none',
            cmap=cmap,
            vmin=0,
            vmax=vmax
        )
        fig.colorbar(a, fraction=0.025, pad=0.04)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    if '/' in layer:
        layer = '2_3'
        
    plt.savefig(sav_dir + filename + '_' + proj_type + '_top_projection_layer_' + layer + kind + '.png', dpi=300)
    plt.savefig(sav_dir + filename + '_' + proj_type + '_top_projection_layer_' + layer + kind + '.eps', dpi=300)
        
            

    tiff.imwrite(sav_dir + filename + '_' + proj_type + '_top_projection_layer_' + layer + kind + '.tif', np.asarray(top_projection_max.T, dtype=np.uint32))


    #%% Project onto butterfly
    
    
    # layer = 'all'
    # template = tiff.imread(input_path + 'downsampled_standard_downsampled_standard.tiff')
    
    
    """ If we want a butterfly map """
    proj_top = ccfproj.Isocortex2dProjector(
        # Specify our view lookup file
        "./extra_downloads/flatmap_butterfly.h5",
    
        # Specify our streamline file
        "./extra_downloads/surface_paths_10_v3.h5",
    
        # Specify that we want to project both hemispheres
        hemisphere="both",
    
        # The butterfly view doesn't contain space for the right hemisphere,
        # but the projector knows where to put the right hemisphere data so
        # the two hemispheres are adjacent if we specify that we're using the
        # butterfly flatmap
        view_space_for_other_hemisphere='flatmap_butterfly',
    )
    
    
    if proj_type=='raw':
        bf_projection_max = proj_top.project_volume(template)
        cmap='Greys_r'
        
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w=8,h=8)
        ax = plt.Axes(fig, [0., 0., 0.9, 0.9])
        ax.set_axis_off()
        fig.add_axes(ax)
    
        
        a = ax.imshow(
            bf_projection_max.T, # transpose so that the rostral/caudal direction is up/down
            interpolation='none',
            cmap=cmap,
        )       
        
        
    elif proj_type=='density' or proj_type=='LARGE_density':
        bf_projection_max = proj_top.project_volume(template, kind=kind)
        cmap = 'turbo'
    
    
        fig = plt.figure(frameon=False)
        fig.set_size_inches(w=8,h=8)
        ax = plt.Axes(fig, [0., 0., 0.9, 0.9])
        ax.set_axis_off()
        fig.add_axes(ax)
    
        
        a = ax.imshow(
            bf_projection_max.T, # transpose so that the rostral/caudal direction is up/down
            interpolation='none',
            cmap=cmap,
            vmin=0,
            vmax=vmax
        )
        fig.colorbar(a, fraction=0.025, pad=0.04)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        
    plt.savefig(sav_dir + filename + '_' + proj_type + '_butterfly_projection_layer_' + layer + kind +'.png', dpi=300)
    plt.savefig(sav_dir + filename + '_' + proj_type + '_butterfly_projection_layer_' + layer + kind +'.eps', dpi=300)
        
            
    
    tiff.imwrite(sav_dir + filename + '_' + proj_type + '_butterfly_projection_layer_' + layer + kind + '.tif', np.asarray(bf_projection_max.T, dtype=np.uint32))
    
    
    
    
    #%% Draw boundaries
    
    # """
    # Since the CCF is an annotated reference atlas, we also know which isocortical region each surface voxel belongs to. 
    # It is often useful to draw the region boundaries on top of the projected images, and we can use the BoundaryFinder 
    # object to do so. (We’ll need the appropriate atlas files from Data Files to do this, too.)
    # """
    # bf_boundary_finder = ccfproj.BoundaryFinder(
    #     projected_atlas_file="./extra_downloads/flatmap_butterfly.nrrd",
    #     labels_file="./extra_downloads/labelDescription_ITKSNAPColor.txt",
    # )
    
    # # We get the left hemisphere region boundaries with the default arguments
    # bf_left_boundaries = bf_boundary_finder.region_boundaries()
    
    # # And we can get the right hemisphere boundaries that match up with
    # # our projection if we specify the same configuration
    # bf_right_boundaries = bf_boundary_finder.region_boundaries(
    #     # we want the right hemisphere boundaries, but located in the right place
    #     # to plot both hemispheres at the same time
    #     hemisphere='right_for_both',
    
    #     # we also want the hemispheres to be adjacent
    #     view_space_for_other_hemisphere='flatmap_butterfly',
    # )
    
    # """
    # These boundaries are returned as dictionaries with the region acronyms as the keys and the values as
    #  2D arrays of the boundary coordinates (in the space of the projection).
    # """
    
    
    # bf_left_boundaries
    
    # """
    # Now we can plot them on top of the average template projection.
    # """

    # #fig = plt.figure()
    # plt.imshow(
    #     bf_projection_max.T,
    #     interpolation='none',
    #     cmap='Greys_r',
    # )




    # im_poly = np.zeros(np.shape(bf_projection_max.T))
    # im_bounds = np.zeros(np.shape(bf_projection_max.T))
    
    # item_num = 1
    # for k, boundary_coords in bf_left_boundaries.items():
    #     plt.plot(*boundary_coords.T, c="white", lw=0.5)
        
    #     # draw polygon into numpy array
    #     rr, cc = skimage.draw.polygon(boundary_coords[:, 0], boundary_coords[:, 1])
    #     im_poly[cc, rr] = item_num
    #     item_num += 1

                
    #     # draw text onto plot
    #     plt.text(np.mean(rr), np.mean(cc), k, fontsize=6)
        

    #     ### draw boundaries of array
    #     rr, cc = skimage.draw.polygon_perimeter(boundary_coords[:, 0], boundary_coords[:, 1])
    #     im_bounds[cc, rr] = 255

        

    # item_num = 1
    # for k, boundary_coords in bf_right_boundaries.items():
    #     print(k)
    #     plt.plot(*boundary_coords.T, c="white", lw=0.5)

    #     # draw polygon into numpy array
    #     rr, cc = skimage.draw.polygon(boundary_coords[:, 0], boundary_coords[:, 1])
    #     im_poly[cc, rr] = item_num
    #     item_num += 1
        
    #     # draw text onto plot
    #     plt.text(np.mean(rr), np.mean(cc), k, fontsize=6)
        

    #     ### draw boundaries of array       
    #     rr, cc = skimage.draw.polygon_perimeter(boundary_coords[:, 0], boundary_coords[:, 1])
    #     im_bounds[cc, rr] = 255  
    

    # # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.axis('off')
    # plt.savefig(sav_dir + '_fig' + layer + '.png', dpi=300)
    
    # tiff.imwrite(sav_dir + filename + '_' + proj_type + '_butterfly_polygons' + layer + '.tif', np.asarray(im_poly, dtype=np.uint8))
    # tiff.imwrite(sav_dir + filename + '_' + proj_type + '_butterfly_bounds' + layer + '.tif', np.asarray(im_bounds, dtype=np.uint8))
    






    
    