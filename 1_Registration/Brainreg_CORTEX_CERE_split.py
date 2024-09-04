# -*- coding: utf-8 -*-

import glob, os

import numpy as np
import matplotlib.pyplot as plt
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import tifffile as tiff
import z5py

from brainreg.cli import main as brainreg_run
import sys
from postprocess_func import *

from skimage import morphology
from skimage.util import img_as_uint
from numpy import inf

import sys
sys.path.append("..")

from get_brain_metadata import *

                    

#%% List of brains
# list_brains = get_metadata(mouse_num = ['M271', 'M312', 'M265'])
list_brains = get_metadata(mouse_num = 'all')

# list_brains = get_metadata(mouse_num = ['Otx18'])

# list_brains = get_metadata(mouse_num = ['M285', 'M97', 'M147', 'M155', 'M152', 'M146', 'M170', 'M172'])


# list_brains = get_metadata(mouse_num = ['M312', 'M313', 'M310'])

# list_brains = get_metadata(mouse_num = ['M310','M312', 'M313'])


# list_brains = get_metadata(mouse_num = ['5Otx5', 'Otx6'])
# list_brains = get_metadata(mouse_num = ['M265', 'M266', 'M267'])   ### dont bother using full autofluor brain, doesnt get rid of cerebellum as nicely

# list_brains = get_metadata(mouse_num = ['M265', 'M266', 'M267', 'M310', 'M311', 'M312', 'M313']) 
# ch_auto = 'ch1'   ### if want to use 633



# P60
# list_brains = get_metadata(mouse_num = ['M127', 'M229', 'M126', 'M299', 'M254', 'M256', 'M260', 'M223'])
list_brains = get_metadata(mouse_num = ['M169'])
ch_auto = 'ch0'   ### if want to use myelin for registration


XY_res = 1.152035240378141
Z_res = 5
num_channels = 2

n4_bool = 1
mask = 1
scale = 1

resolution = 20   ### for cortex reg

add_myelin = 0

divide_myelin = 1


stripefilt = 1

pad = True   

if scale:
    z_after = z_before = 50
    x_after = x_before = 50
    
else:
    z_after = z_before = 50
    x_after = x_before = 50




for input_dict in list_brains:
    
    input_path = input_dict['path']
    thresh_div = input_dict['thresh']
    exp_name = input_dict['name']
    
    
    sav_dir = input_path + '/' + exp_name + '_TIFFs_extracted/' + exp_name + '_ANTS_registered/'

    
    # ch_auto = input_dict['ch_auto']
    
    ### Only divide by myelin if thresh is set!!!
    if thresh_div > 0:
        divide_myelin = 1
        if ch_auto != 'ch0':
            ch_auto = 'ch2'
        
    elif thresh_div == 0:
        divide_myelin = 0
    
    
    if divide_myelin:
        atlas = 'allen_mouse_CORTEX_20um'        
        if resolution == 10:
            atlas = 'allen_mouse_10um'
            
        ### register using myelin!
        if ch_auto == 'ch0':    
            atlas = 'allen_mouse_MYELIN_20um'
            
            
            
    else:
    # divide_myelin = 0
        atlas = 'allen_mouse_20um_CUBIC'   ### not the fullbrain!
   


    #%% Run BrainReg
    print('Starting BrainReg')
    
    ### Register using myelin
    if ch_auto == 'ch0':
        reg_name = sav_dir + exp_name + '_ANTS_myelin_cortex_only.tif'
    
        # mask = sav_dir + exp_name + '_ANTS_to_remove.tif'
        add_chs = []
        add_chs.append(sav_dir + exp_name + '_ANTS_cortex_only.tif')
    
    else:   ### Register using autofluorescence channel
        reg_name = sav_dir + exp_name + '_ANTS_cortex_only.tif'
        
        # mask = sav_dir + exp_name + '_ANTS_to_remove.tif'
        add_chs = []
        add_chs.append(sav_dir + exp_name + '_ANTS_myelin_cortex_only.tif')
          

    
    #%% If want to also do add channel with NO myelin DIV (because forgot to do it during ANTS)
    if divide_myelin:
        autofluor_path = input_path + '/' + exp_name + '_TIFFs_extracted/' + exp_name + '_level_s3_ch1_n4_down1_resolution_' + str(resolution) + '_PAD.tif'
        autofluor_raw = tiff.imread(autofluor_path)
        autofluor_raw = np.flip(autofluor_raw, axis=2)
        autofluor_raw = np.flip(autofluor_raw, axis=0)  ### flip the Z-axis
        autofluor_raw = np.moveaxis(autofluor_raw, 0, 1)   ### reshuffle atlas so now in proper orientation
    
        ### also read in the atlas to mask out
        atlas_im= tiff.imread(sav_dir + exp_name + '_ANTS_to_remove.tif')
        lower_thresh = 2000
        autofluor_raw[atlas_im > 0] = 0    
        autofluor_raw[autofluor_raw < lower_thresh] = 0
    
        save_path = input_path + '/' + exp_name + '_TIFFs_extracted/' + exp_name + '_level_s3_ch1_n4_down1_resolution_' + str(resolution) + '_PAD_ROTATED_CORTEX_ONLY.tif'
        tiff.imwrite(save_path, autofluor_raw)
    
        ### OR --- dont read in atlas to mask out! --- this is truly just to achieve the mean autofluor with full body (and cerebellum)
        ### but this doesnt work... because there's no actual registration of the cerebellum...
        # lower_thresh = 2000 
        # autofluor_raw[autofluor_raw < lower_thresh] = 0
    
        # save_path = input_path + '/' + exp_name + '_TIFFs_extracted/' + exp_name + '_level_s3_ch1_n4_down1_resolution_' + str(resolution) + '_PAD_ROTATED_FULL_BRAIN.tif'
        # tiff.imwrite(save_path, autofluor_raw)    
        
        
    
    
    # if not divide_myelin:
    # #     ### use this as the registration channel!
    #     reg_name = save_path
        
    # else:  ### otherwise just include it as an additional channel during registration
        add_chs.append(save_path)    # get all tiffs saved from above

    

    
    ### SCALING
    whole_brain_voxel_sizes = (str(20), str(20), str(20))
    grid_spacing = '-10'     ### worse with -5 than -10
    bending_energy = '0.9'  # was 0.95, testing with expanded brains was best with 0.5?
    
    # orientation = 'ial'
    
    orientation = 'asr' 
    
    
    num_cpus = '8'
    
    smooth_gauss = '0'   ### default -1 *** which is in voxels!!!
    use_steps = 'default'
    
    
    whole_brain_data_dir = sav_dir + exp_name + atlas + '_CORTEX_ONLY_DIVIDE_MYELIN_' + bending_energy + '_n4_' + str(n4_bool) + '_grid_' + grid_spacing + '_gauss_' + smooth_gauss  + '_use_steps_' + str(use_steps)
   
    try:
        # Create target Directory
        os.mkdir(whole_brain_data_dir)
        print("\nSave directory " , whole_brain_data_dir ,  " Created ") 
    except FileExistsError:
        print("\nSave directory " , whole_brain_data_dir ,  " already exists")
    whole_brain_data_dir = whole_brain_data_dir + '/'
    
    brainreg_args = [
        "brainreg", str(reg_name), str(whole_brain_data_dir),
        "-v",
        whole_brain_voxel_sizes[0],
        whole_brain_voxel_sizes[1],
        whole_brain_voxel_sizes[2],
        "--orientation", orientation,
        "--n-free-cpus", num_cpus,
        "--atlas", atlas,
        #"--debug",  ### save some space
        "--grid-spacing", grid_spacing, ### negative value in voxels, positive in mm, smaller means more local deforms  default: -10
    
        "--bending-energy-weight", bending_energy, ### between 0 and 1 (default: 0.95)
    
        "--smoothing-sigma-reference", '0', ### adds gaussian to reference image # default 1 is bad
        "--smoothing-sigma-floating", smooth_gauss,  ### adds gaussian to moving image # default 1 is bad
    ]
    
    ### Add additional channels
    if len(add_chs) > 0:
        brainreg_args.append("--additional")
        for name in add_chs:
            brainreg_args.append(name)
    ## Run Brainreg
    sys.argv = brainreg_args
    brainreg_run()    

    ### save numpy array with all the params tested, or just a string
    if pad:
        brainreg_args.append(z_after)
        brainreg_args.append(x_after)
    np.save(whole_brain_data_dir + 'brainreg_args.npy', brainreg_args)
    







    #%% Start BrainReg for Cerebellum

    # print('Starting Cerebellum BrainReg')
    
    # atlas = 'allen_mouse_CERE_20um'   
    # # if ch_auto == 'ch1':
    # #     atlas = 'allen_mouse_CERE_20um'        
    # #     if resolution == 10:
    # #         atlas = 'allen_mouse_CORTEX_10um'
    
    
    # # ### Register with plain autofluor --- no division by myelin
    # # elif ch_auto == 'no_div':
    # #         atlas = 'allen_mouse_20um_CUBIC'
    # #         zzz
    # #     # zzz
        
    # # ### Register with myelin
    # # elif ch_auto == 'ch0':
    # #     atlas = 'allen_mouse_MYELIN_20um'
    # #     zzz
    


    
    
    # ### Register using myelin
    # # if ch_auto == 'ch0' and divide_myelin:
    # #     reg_name = sav_dir + exp_name + '_ANTS_myelin_cerebellum_only.tif'
    
    # #     # mask = sav_dir + exp_name + '_ANTS_to_remove.tif'
    # #     add_chs = []
    # #     add_chs.append(sav_dir + exp_name + '_ANTS_cerebellum_only.tif')
    
    # # else:   ### Register using autofluorescence channel
    # reg_name = sav_dir + exp_name + '_ANTS_cerebellum_only.tif'
    
    # # mask = sav_dir + exp_name + '_ANTS_to_remove.tif'
    # add_chs = []
    # add_chs.append(sav_dir + exp_name + '_ANTS_myelin_cerebellum_only.tif')
      

    
    # #%% If want to also do add channel with NO myelin DIV (because forgot to do it during ANTS)
    
    # autofluor_path = input_path + '/' + exp_name + '_TIFFs_extracted/' + exp_name + '_level_s3_ch1_n4_down1_resolution_' + str(resolution) + '_PAD.tif'
    # autofluor_raw = tiff.imread(autofluor_path)
    # autofluor_raw = np.flip(autofluor_raw, axis=2)
    # autofluor_raw = np.flip(autofluor_raw, axis=0)  ### flip the Z-axis
    # autofluor_raw = np.moveaxis(autofluor_raw, 0, 1)   ### reshuffle atlas so now in proper orientation

    # ### also read in the atlas to mask out
    # atlas_im = tiff.imread(sav_dir + exp_name + '_ANTS_to_remove.tif')
    # lower_thresh = 2000
    # autofluor_raw[atlas_im == 0] = 0    
    # autofluor_raw[autofluor_raw < lower_thresh] = 0

    # save_path = input_path + '/' + exp_name + '_TIFFs_extracted/' + exp_name + '_level_s3_ch1_n4_down1_resolution_' + str(resolution) + '_PAD_ROTATED_CEREBELLUM_ONLY.tif'
    # tiff.imwrite(save_path, autofluor_raw)
   
    
    # if not divide_myelin:
    #     ### use this as the registration channel!
    #     reg_name = save_path
        
    # else:
    #     add_chs.append(save_path)    # get all tiffs saved from above


    # ### SCALING
    # whole_brain_voxel_sizes = (str(20), str(20), str(20))
    # grid_spacing = '-10'     ### worse with -5 than -10
    # bending_energy = '0.9'  # was 0.95, testing with expanded brains was best with 0.5?
    
    # # orientation = 'ial'
    
    # orientation = 'asr' 
    
    
    # num_cpus = '8'
    
    # smooth_gauss = '0'   ### default -1 *** which is in voxels!!!
    # use_steps = 'default'
    
    
    # whole_brain_data_dir = sav_dir + exp_name + atlas + 'CEREBELLUM_ONLY_DIVIDE_MYELIN_' + bending_energy + '_n4_' + str(n4_bool) + '_grid_' + grid_spacing + '_gauss_' + smooth_gauss  + '_use_steps_' + str(use_steps)
   
    # try:
    #     # Create target Directory
    #     os.mkdir(whole_brain_data_dir)
    #     print("\nSave directory " , whole_brain_data_dir ,  " Created ") 
    # except FileExistsError:
    #     print("\nSave directory " , whole_brain_data_dir ,  " already exists")
    # whole_brain_data_dir = whole_brain_data_dir + '/'
    
    # brainreg_args = [
    #     "brainreg", str(reg_name), str(whole_brain_data_dir),
    #     "-v",
    #     whole_brain_voxel_sizes[0],
    #     whole_brain_voxel_sizes[1],
    #     whole_brain_voxel_sizes[2],
    #     "--orientation", orientation,
    #     "--n-free-cpus", num_cpus,
    #     "--atlas", atlas,
    #     #"--debug",  ### save some space
    #     "--grid-spacing", grid_spacing, ### negative value in voxels, positive in mm, smaller means more local deforms  default: -10
    
    #     "--bending-energy-weight", bending_energy, ### between 0 and 1 (default: 0.95)
    
    #     "--smoothing-sigma-reference", '0', ### adds gaussian to reference image # default 1 is bad
    #     "--smoothing-sigma-floating", smooth_gauss,  ### adds gaussian to moving image # default 1 is bad
    # ]
    
    # ### Add additional channels
    # if len(add_chs) > 0:
    #     brainreg_args.append("--additional")
    #     for name in add_chs:
    #         brainreg_args.append(name)
    # ## Run Brainreg
    # sys.argv = brainreg_args
    # brainreg_run()    

    # ### save numpy array with all the params tested, or just a string
    # if pad:
    #     brainreg_args.append(z_after)
    #     brainreg_args.append(x_after)
    # np.save(whole_brain_data_dir + 'brainreg_args.npy', brainreg_args)
    
            




            