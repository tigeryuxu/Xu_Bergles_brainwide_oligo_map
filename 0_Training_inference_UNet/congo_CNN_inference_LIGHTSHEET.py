# -*- coding: utf-8 -*-
"""
Seg-CNN:
    Expected input:
        - series of Tiffs
        
        
        - need GPU with at least 6 GB RAM
        
        

"""

import sys
sys.path.insert(0, './layers')

import sys,os
sys.path.append('/home/user/build/SimpleITK-build/Wrapping/Python/')

from csbdeep import io

from keras.models import load_model
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
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

import torch
#from UNet_pytorch_online import *
from layers.tracker import *

from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.UNet_functions_PYTORCH import *
from functional.GUI import *
import tifffile as tiff

from layers.UNet_pytorch_online import *

from skimage.filters import threshold_otsu
from skimage.filters import threshold_triangle
from skimage.transform import rescale, resize, downscale_local_mean

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True

import pandas as pd

""" Define GPU to use """
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

"""  Network Begins: """
#s_path = './(21) Checkpoints_PYTORCH_NO_transforms_AdamW_batch_norm_CLEAN_DATA_LARGE_NETWORK/'
# s_path = './Checkpoints/'
# s_path = '/media/user/FantomHD1/Lightsheet data/Checkpoints_lightsheet/(2) Check_lightsheet_NO_transforms_AdamW_batch_norm_5deep/'
# s_path = '/media/user/FantomHD1/Lightsheet data/Checkpoints_lightsheet/(4) Check_lightsheet_NO_transforms_5deep_FIXED_SCALING_NEW_DATA/'

s_path = '/media/user/4TB_SSD/Training_UNet_Congo/(1) Check_lightsheet_NO_transforms_4deep_LR_same/'

# overlap_percent = 0
# input_size = 256
# depth = 64
# num_truth_class = 2


### NEW TRAINED SEGCNN
overlap_percent = 0
input_size = 128
depth = 16
num_truth_class = 2

XY_expected = 0.83; Z_expected = 3;

""" TO LOAD OLD CHECKPOINT """
onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
onlyfiles_check.sort(key = natsort_key1)


last_file = onlyfiles_check[-1]
split = last_file.split('check_')[-1]
num_check = split.split('.')
checkpoint = num_check[0]
checkpoint = 'check_' + checkpoint

print('restoring weights from: ' + checkpoint)
check = torch.load(s_path + checkpoint, map_location=lambda storage, loc: storage)
tracker = check['tracker']

unet = check['model_type']; 
""" Initialize network """  
# kernel_size = 5
# pad = int((kernel_size - 1)/2)
# unet = UNet_online(in_channels=1, n_classes=2, depth=5, wf=4, kernel_size = kernel_size, padding= int((kernel_size - 1)/2), 
#                     batch_norm=True, batch_norm_switchable=False, up_mode='upsample')

unet.load_state_dict(check['model_state_dict'])
unet.to(device); unet.eval()
print('parameters:', sum(param.numel() for param in unet.parameters()))


""" Select multiple folders for analysis AND creates new subfolder for results output """
list_folder, XY_res, Z_res = seg_CNN_GUI()


XY_scale = float(XY_res)/XY_expected

""" Loop through all the folders and do the analysis!!!"""
for input_path in list_folder:
    foldername = input_path.split('/')[-2]
    sav_dir = input_path + '/' + foldername + '_segCNN'

    """ For testing ILASTIK images """
    images = glob.glob(os.path.join(input_path,'*00.h5'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('00.h5','.xml'), ilastik=i.replace('.tif','_single_Object Predictions_.tiff')) for i in images]
     
    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("\nSave directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("\nSave directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    
    # Required to initialize all
    for i in range(len(examples)):
         
         
         """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
         with torch.set_grad_enabled(False):  # saves GPU RAM            
            input_name = examples[i]['input']  
            # input_im = tiff.imread(input_name)
            
            
            import h5py
            f = h5py.File(examples[i]['input'], "r")
            
            print(f.keys())
            print(f['s00'].keys())
            print(f['t00000'].keys())
            print(f['t00000']['s00'].keys())
            
            #lowest_res = f['t00000']['s00']['7']['cells']
            highest_res = f['t00000']['s00']['0']['cells']
            
            dset = highest_res
            
            ### channel 2
            #highest_res = f['t00000']['s01']['0']['cells']
            
            
            coords_df = pd.DataFrame(columns = ['offset', 'block_num', 'Z', 'X', 'Y', 'Z_scaled', 'X_scaled', 'Y_scaled', 'equiv_diam', 'vol'])
            
            

            """ Or save to memmapped TIFF first... """
            print('creating memmap save file on disk')
            #memmap_save = tiff.memmap(sav_dir + 'temp_SegCNN.tif', shape=dset.shape, dtype='uint8')
            
            memmap_save = tiff.memmap('/media/user/storage/Temp_lightsheet/temp_SegCNN_' + str(i) + '_.tif', shape=dset.shape, dtype='uint8')


            """ Figure out how many blocks need to be generated first, then loop through all the blocks for analysis """
            print('Extracting block sizes')
            
            """ Or use chunkflow instead??? https://pychunkflow.readthedocs.io/en/latest/tutorial.html"""
            im_size = np.shape(dset);
            depth_im = im_size[0];  height = im_size[1];  width = im_size[2]; 
            
            total_blocks = 0;
            all_xyz = []                                               
             
            
            
            """ These should be whole values relative to the chunk size so that it can be uploaded back later! """
            # quad_size = 128 * 10
            # quad_depth = 64 * 4
            
            quad_size = 128 * 14
            quad_depth = 64 * 4
            
            # quad_size = round(input_size * 1/XY_scale * 3)
            # quad_depth = round(depth * 1/XY_scale * 3)
            
            #overlap_percent = 0
            
                
            for x in range(0, width + quad_size, round(quad_size - quad_size * overlap_percent)):
                  if x + quad_size > width:
                       difference = (x + quad_size) - width
                       x = x - difference
                            
                  for y in range(0, height + quad_size, round(quad_size - quad_size * overlap_percent)):
                       
                       if y + quad_size > height:
                            difference = (y + quad_size) - height
                            y = y - difference
                       
                       for z in range(0, depth_im + quad_depth, round(quad_depth - quad_depth * overlap_percent)):
                           #batch_x = []; batch_y = [];
                 
                           if z + quad_depth > depth_im:
                                difference = (z + quad_depth) - depth_im
                                z = z - difference
                           
                               
                           """ Check if repeated """
                           skip = 0
                           for coord in all_xyz:
                                if coord == [z, x, y]:
                                     skip = 1
                                     break                      
                           if skip:  continue
                                
                           print([z, x, y])
                           all_xyz.append([z, x, y])
                           
            
            
            ### how many total blocks to analyze:
            print(len(all_xyz))
                
            
            
            
            """ Then loop through """
            for id_c, s_c in enumerate(all_xyz):
                
            
            ### for continuing the run
            #for id_c in range(93, len(all_xyz)):
                s_c = all_xyz[id_c]
                
                ### for debug:
                #s_c = all_xyz[10]
                
                
                import time
                tic = time.perf_counter()
                
                
                ### while debugging:
                #s_c = [0, 0, 5544]
                    
                #input_im =  dset[200:400,5000:6000,5000:6000]          
                
                input_im = dset[s_c[0]:s_c[0] + quad_depth, s_c[1]:s_c[1] + quad_size, s_c[2]:s_c[2] + quad_size];  
                og_shape = input_im.shape
                """ Convert to uint8 if not """
                # if input_im.dtype != np.uint8:
                #     #input_im = np.asarray(input_im, dtype=np.uint8)
                #     #new_max = 1200
                #     #new_max = 3000
                #     new_max = 8000
                #     input_im[input_im > new_max] = new_max

                #     im_norm = (input_im/new_max) * 255                
                    
                #     input_im = np.asarray(im_norm, dtype=np.uint8)
            
                # """ also detect if blank (less than 1000 voxels with value > 40) """
                
                # if len(np.where(input_im > 40)[0]) < 1000:
                #     print('skipping: ' + str(s_c))
                #     continue
            
            
                """ Detect if blank in uint16 """
                num_voxels = len(np.where(input_im > 300)[0])
                if num_voxels < 10000:
                     print('skipping: ' + str(s_c))
                     print('num_blank: ' + str(num_voxels))
                     continue                
            
            
                
                # import napari
                # d = napari.view_image(input_im)
                # d.add_image(input_im)        
                                    
                print('Analyzing: ' + str(s_c))
                print('Which is: ' + str(id_c) + ' of total: ' + str(len(all_xyz)))
                        
                
            
                
            

                """ Scale images to default resolution if user resolution does not matching training """
                
                # if XY_scale < 1: print('Image XY resolution does not match training resolution, and will be downsampled by: ' + str(round(1/XY_scale, 2)))
                # elif  XY_scale > 1: print('Image XY resolution does not match training resolution, and will be upsampled by: ' + str(round(XY_scale, 2)))
    
    
                # Z_scale = float(Z_res)/Z_expected
                # if Z_scale < 1: print('Image Z resolution does not match training resolution, and will be downsampled by: ' + str(round(1/Z_scale, 2)))
                # elif  Z_scale > 1: print('Image Z resolution does not match training resolution, and will be upsampled by: ' + str(round(Z_scale, 2)))
                
                
                # if XY_scale != 1 or Z_scale != 1:
                #     input_im = rescale(input_im, [Z_scale, XY_scale, XY_scale], anti_aliasing=True, order=2, preserve_range=True)   ### rescale the images
                        
                #     input_im = ((input_im - input_im.min()) * (1/(input_im.max() - input_im.min()) * 255)).astype('uint8')   ### rescale to 255
    
    
                """ Analyze each block with offset in all directions """
                ### CATCH error if too small volume, need to pad with zeros!!!
                if input_im.shape[0] < depth: pad_z = depth 
                else: pad_z = input_im.shape[0] 
    
                if input_im.shape[1] < input_size: pad_x = input_size 
                else: pad_x = input_im.shape[1] 
                
                if input_im.shape[2] < input_size: pad_y = input_size 
                else: pad_y = input_im.shape[2]             
                
                pad_im = np.zeros([pad_z, pad_x, pad_y])
                pad_im[:input_im.shape[0], :input_im.shape[1], :input_im.shape[2]] = input_im
                input_im = pad_im
                #input_im = np.asarray(input_im, dtype=np.uint8)
                
                
                
                
                
                """ Find reference free SNR """
                # all_SNR = [];        
                # thresh = threshold_otsu(input_im)
                # for slice_depth in range(0, len(input_im) - 33, 33):
                
                #     first_slices= input_im[slice_depth:slice_depth + 33, ...]
                #     max_first = plot_max(first_slices, ax=0, plot=0)
                #     signal = np.mean(np.where(max_first > thresh))
                #     noise = np.std(np.where(max_first < thresh))
                #     SNR = 10 * math.log10(signal/noise)
                #     all_SNR.append(round(SNR, 3))
                # all_SNR = np.asarray(all_SNR)
                # below_thresh_SNR = np.where(all_SNR < 1.5)[0]
                # if len(below_thresh_SNR) > 0:
                #     print('\nWARNING: SNR is low for image: ' + input_name + 
                #           '\n starting at depth slice: ' + str(below_thresh_SNR * 33) + 
                #           '\n with SNR values: ' + str(all_SNR[below_thresh_SNR]) )
                
                
                
                
                toc = time.perf_counter()
                
                print(f"Opened subblock in {toc - tic:0.4f} seconds")
                

                
                
                
                
                
            
                """ Start inference on volume """
                tic = time.perf_counter()
                
                
                import warnings
                overlap_percent = 0.1
                
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    print('\nStarting inference on volume: ' + str(id_c + 1) + ' of total: ' + str(len(all_xyz)))
                    segmentation = UNet_inference_by_subparts_PYTORCH(unet, device, input_im, overlap_percent, quad_size=input_size, quad_depth=depth,
                                                          mean_arr=tracker.mean_arr, std_arr=tracker.std_arr, num_truth_class=num_truth_class,
                                                          skip_top=1, batch_size=20)
               
                
                toc = time.perf_counter()
                
                print(f"Inference in {toc - tic:0.4f} seconds")
                
                tic = time.perf_counter()

                segmentation[segmentation > 0] = 255
                
                segmentation = np.asarray(segmentation, dtype=np.uint8)   ### convert to uint8
                filename = input_name.split('/')[-1].split('.')[0:-1]
                filename = '.'.join(filename)
                
                ### if operating system is Windows, must also remove \\ slash
                if os_windows:
                     filename = filename.split('\\')[-1]


                # import napari
                # d = napari.view_image(input_im)
                # d.add_image(segmentation)        
                                    

                # plot_max(input_im)
                # plot_max(segmentation); plt.pause(0.1)
                
                
                
                #zzz
                
                 
                tiff.imsave(sav_dir + filename + '_' + str(int(id_c)) +'_segmentation.tif', segmentation)
                               
                #input_im = np.asarray(input_im, np.uint8)
                tiff.imsave(sav_dir + filename + '_' + str(int(id_c)) +'_input_im.tif', np.asarray(input_im, dtype=np.uint16))
                

                                
                

                """ Rescale back to size of original file!!! """                
                # if XY_scale != 1 or Z_scale != 1:
                #     #input_im = rescale(input_im, [Z_scale, XY_scale, XY_scale], anti_aliasing=True, order=2, preserve_range=True)   ### rescale the images
                    
                    
                #     segmentation = resize(segmentation, og_shape)
                #     segmentation[segmentation > 0] = 255
                #     segmentation = np.asarray(segmentation, dtype=np.uint8)
                    
                #     #input_im = ((input_im - input_im.min()) * (1/(input_im.max() - input_im.min()) * 255)).astype('uint8')   ### rescale to 255
                
                
                """ Output to neuroglancer precomputed file """
                
                #vol[s_c[0]:s_c[0] + quad_depth, s_c[1]:s_c[1] + quad_size, s_c[2]:s_c[2] + quad_size, 0] = np.expand_dims(segmentation[0:quad_depth, 0:quad_size, 0:quad_size], axis=-1)
                
                toc = time.perf_counter()
                
                print(f"Rescale up {toc - tic:0.4f} seconds")                
                
                """ Output to memmmapped arr """
                
                tic = time.perf_counter()
                memmap_save[s_c[0]:s_c[0] + quad_depth, s_c[1]:s_c[1] + quad_size, s_c[2]:s_c[2] + quad_size] = segmentation
                memmap_save.flush()
                
                toc = time.perf_counter()
                
                print(f"Save in {toc - tic:0.4f} seconds")
                
                
                """ Also save list of coords of where the cells are located so that can easily access later (or plot without making a whole image!) """
                print('saving coords')
                tic = time.perf_counter()
                from skimage import measure
                labels = measure.label(segmentation)
                #blobs_labels = measure.label(blobs, background=0)
                cc = measure.regionprops(labels)
                
                
                ########################## TOO SLOW TO USE APPEND TO ADD EACH ROW!!!
                # for cell in cc:
                #     center = cell['centroid']
                #     center = [round(center[0]), round(center[1]), round(center[2])]
                    
                #     row = {'offset': s_c, 'block_num': id_c, 
                #            'Z': center[0], 'X': center[1], 'Y': center[2],
                #            'Z_scaled': center[0] + s_c[0], 'X_scaled': center[1] + s_c[1], 'Y_scaled': center[2] + s_c[2]}
                #     coords_df = coords_df.append(row, ignore_index = True)                
                        
                # np.save(sav_dir + filename + '_numpy_arr', coords_df)

                
                ######################### MUCH FASTER TO JUST MAKE A DICTIONARY FIRST, AND THEN CONVERT TO DATAFRAME AND CONCAT
                d = {}
                for i_list, cell in enumerate(cc):
                    center = cell['centroid']
                    center = [round(center[0]), round(center[1]), round(center[2])]
                    
                    d[i_list] = {'offset': s_c, 'block_num': id_c, 
                           'Z': center[0], 'X': center[1], 'Y': center[2],
                           'Z_scaled': center[0] + s_c[0], 'X_scaled': center[1] + s_c[1], 'Y_scaled': center[2] + s_c[2],
                           'equiv_diam': cell['equivalent_diameter'], 'vol': cell['area']}
                
                
                
                df = pd.DataFrame.from_dict(d, "index")
        
                coords_df = pd.concat([coords_df, df])           
                                        
                
                toc = time.perf_counter()
                
                
                np.save(sav_dir + filename + '_numpy_arr', coords_df)
                print(f"Save coords in {toc - tic:0.4f} seconds")

        
    print('\n\nSegmented outputs saved in folder: ' + sav_dir)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    