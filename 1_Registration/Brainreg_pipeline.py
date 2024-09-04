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

from brainreg.cli import main as brainreg_run
import sys
from postprocess_func import *

from skimage import morphology
from skimage.util import img_as_uint
from numpy import inf

import sys
sys.path.append("..")

from get_brain_metadata import *



import imagej
ij = imagej.init('/home/user/Documents/fiji-linux64_NEW_FOR_BIG_STITCHER/Fiji.app', mode='interactive')

                    
def div_by_myel(myelin_im, autofluor, thresh=500):
    #thresh = 500   ### 700 for older brains

    binary = myelin_im > thresh

    # Then despeckle
    b = morphology.remove_small_objects(binary, 100)
    
    # Then mask out
    masked = np.copy(myelin_im)
    masked[b == 0] = 0
    
    
    # Then do gaussian
    #from skimage.filters import gaussian
    #masked_filt = gaussian(masked, sigma=0.5)
    

    # convert back to uint16
    final_im = img_as_uint(masked)

    ### scale division to be within reasonable values
    #final_im[final_im > 2] = 2
    extracted_myelin = np.copy(final_im)
    #tiff.imwrite(input_path + 'myelin_extracted.tif', final_im)
    
    ### scale final im so not dividing by too much
    final_im = (final_im/np.max(final_im)) 
    final_im[masked > 0] = final_im[masked > 0] + 2  ### increased to 2 instead of just 1 fold
    
    ### Divide autofluor im
    div_im = autofluor/final_im   ### avoid divide by 0
    div_im[div_im == inf] = autofluor[div_im == inf] 
    div_im = np.asarray(div_im, dtype=np.uint16)
    
    return div_im, extracted_myelin
    


                    
################## for 633



#%% List of brains
# list_brains = get_metadata(mouse_num = ['M271', 'M312', 'M265'])
# list_brains = get_metadata(mouse_num = 'all')

list_brains = get_metadata(mouse_num = ['M115'])


# list_brains = get_metadata(mouse_num = ['M265', 'M266', 'M267', 'M310', 'M311', 'M312', 'M313', 'M248', 'M246']) 


ch_auto = 'ch1'   ### if want to use 633
XY_res = 1.152035240378141
Z_res = 5
num_channels = 2

n4_bool = 1
mask = 1
scale = 1

resolution = 20   ### for cortex reg
# resolution = 20   ### for more rapid simple registration

add_myelin = 0

divide_myelin = 1

stripefilt = 1


################## for 488
# list_paths = [
#             #'/media/user/Tx_LS_Data_7small/20231217_M239_WT_with_DAPI_488_638_405_561_P56_delip_RIMS_RI_1489_sunflower_80perc/fused/'

#             '/media/user/20TB_HDD_NAS/20240127_M290_WT_P60_SHIELD_delip8d_RIMS_RI_1493_sunflow_laser_80perc/'

#                 ]
# ch_auto = 'ch0'   ### if want to use 488
# XY_res = 1.8432563846050258
# Z_res = 8
# num_channels = 2

# n4_bool = 1
# mask = 1
# scale = 1

# add_myelin = 0

# stripefilt = 1

#############################################################################

level = 's3'   ### level s2 is about 25 GB
pad = True   

#if n4_bool:
# z_after = z_before = 20
# x_after = x_before = 20

# else:
if scale:
    z_after = z_before = 50
    x_after = x_before = 50
    
else:
    z_after = z_before = 50
    x_after = x_before = 50




for input_dict in list_brains:
    
    input_path = input_dict['path']
    thresh_div = input_dict['thresh']
    
    
    ### Only divide by myelin if thresh is set!!!
    if thresh_div > 0:
        divide_myelin = 1
        
    elif thresh_div == 0:
        divide_myelin = 0
    

    images = glob.glob(os.path.join(input_path,'*.n5'))    # can switch this to "*truth.tif" if there is no name for "input"
    
    
    input_name = images[0]  
    filename = input_name.split('/')[-1].split('.')[0:-1]
    filename = '.'.join(filename)      
    
    sav_dir = input_path + '/' + filename + '_TIFFs_extracted'
    try:
        # Create target Directory
        os.mkdir(sav_dir)
        print("\nSave directory " , sav_dir ,  " Created ") 
    except FileExistsError:
        print("\nSave directory " , sav_dir ,  " already exists")
        
    sav_dir = sav_dir + '/'
    
    
    #%% Extract downsampled levels of data and autofluor
    ### load data from .n5
      
      
    ### MAYBE USE WITH INSTEAD??? to prevent weirdness between files
    with z5py.File(input_name, "r") as f:
    
        # get size of original image so know how much the level has downsampled the data
        orig_shape = f['setup0/timepoint0/s0'].shape
        
        
        ### Extract all channels at downsampled resolution
        masked_filt = 0
        for ch in range(num_channels):  

                
            
            dset = f['setup' + str(ch) + '/timepoint0/' + level]
            dset = np.asarray(dset, np.uint16)
            
            ### calculate downsampling factors
            down_XY = orig_shape[1]/dset.shape[1]
            down_Z = orig_shape[0]/dset.shape[0]
            
            
            ###First check if file exists already
            if os.path.isfile(sav_dir + filename +'_level_' + level + '_ch' + str(ch) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution) + '_NOPAD.tif'):
                print('File already extracted, skipping')
                if scale:
                    XY_scale = (XY_res * down_XY)/resolution
                    Z_scale = (Z_res * down_Z)/resolution
                continue
                
                #continue
                
            #     dset = tiff.imread(sav_dir + filename +'_level_' + level + '_ch' + str(ch) + '_NOPAD.tif')
            #     dset[dset < 100] = 0   
            #     constant_val = 100  ### these 0s are from fusion from BigStitcher
                
            ### ALSO SET ALL 0 to be 0 --- use this for n4?
            constant_val = 92  ### these 0s are from fusion from BigStitcher
            
            dset[dset == 0] = constant_val
            constant_val = 0
            #dset[dset < 92] = 0
            
            ### for some reason this works best right now...
            constant_val = 92
            dset[dset < 92] = 92  
            
            im = dset

            
            ### for setting all background to zero INCLUDING PADDING
            # constant_val = 0
            # dset[dset < 100] = 0
            
            # binarize
            #from skimage.filters import threshold_otsu

            



            ### scale to 20microns in all dimensions
            if scale:
                XY_scale = (XY_res * down_XY)/resolution
                Z_scale = (Z_res * down_Z)/resolution
                
                im = rescale(dset, (Z_scale, XY_scale, XY_scale), anti_aliasing=True, preserve_range=True)
                
                
     
            # ### MAKE A BINARY MASK FOR SUBTRACTION LATER
            if mask and (ch == 1 or ch_auto == 'ch0'):
                im_bw = np.copy(im)
                mid_slice = im_bw[np.int32(im_bw.shape[0]/2)]
                from skimage.filters import threshold_triangle
                thresh = threshold_triangle(mid_slice) + 5


                ### LIMIT THE THRESHOLD
                if thresh > 95 and 'M115' in filename:
                    thresh = 95
                    
                ### Also limit it if it gets super biased by weird stuff in image
                if thresh > 200:
                    thresh = 100
                    
                
                
                print('Threshold triangle: ' + str(thresh))
                im_bw = im_bw > thresh
                
 
                from skimage import measure, morphology
                labelled = measure.label(im_bw)
                rp = measure.regionprops(labelled)
                # get size of largest cluster
                id_large = np.argmax([i.area for i in rp])
                reg_coords = rp[id_large]['coords']

                im_bw = np.zeros(np.shape(im_bw))
                im_bw[reg_coords[:, 0], reg_coords[:, 1], reg_coords[:, 2]] = 1
                
                constant_val = 0
            
 

            ### To add myelin
            if add_myelin:
                #zzz
                myelin_thresh = 500
                binary = im > myelin_thresh
                # Then despeckle
                from skimage import morphology
                b = morphology.remove_small_objects(binary, 100)
                
                # dilate
                #b_dil = morphology.isotropic_dilation(b, radius=2)
                
                
                # Then mask out
                masked = np.copy(im)
                masked[b == 0] = 0
 
                # Then do gaussian
                from skimage.filters import gaussian
                masked_filt = gaussian(masked, sigma=2)
                
                
                #from skimage.util import img_as_uint
                # convert back to uint16
                masked_filt = np.asarray(masked_filt, dtype=np.int16)
                #final_im = img_as_uint(masked_filt)
                #masked_filt = masked_filt - 400
                masked_filt[masked_filt < 0] = 0
                masked_filt[masked_filt > 400] = 400
                masked_filt = np.asarray(masked_filt, dtype=np.uint16)

                import napari
                
                viewer = napari.Viewer()
                new_layer = viewer.add_image(im_ahe)
                new_layer = viewer.add_image(im)
                    
                    




            ## MASK OUT IM prior to stripe filter
            if mask == 1 and (ch == 1 or ch_auto == 'ch0'):
              im[im_bw == 0] = 0
              
                            


            """ Adjust histogram BEFORE doing strip filter """
            if ch == 1 or ch_auto == 'ch0':
                ### REMOVE OUTLIERS
                im[im > 1000] = 1000
                
                if 'M115' in filename:
                    im[im > 500] = 95
                    im[im > 300] = 300
                    
                # else:
                #     im[im > 500] = thresh
                #     im[im > 280] = 280
                    
                
                im_norm = scale_and_convert_to_16_bits(im)

                #norm_N4 = N4_correction(im_norm, shrinkFactor=1, numberFittingLevels=4)
                #im = np.asarray(im, dtype=np.uint16)
                from skimage import exposure, util
                #im_test = np.copy(im)
                

                
                ### apply AHE
                im_ahe = exposure.equalize_adapthist(im_norm)
                
                im = scale_and_convert_to_16_bits(im_ahe)
                
                im_tmp = np.copy(im)
                
                

                                


# # im.shape
# import napari

# viewer = napari.Viewer()
# new_layer = viewer.add_image(test)
# # new_layer = viewer.add_image(im_norm)
# # new_layer = viewer.add_image(im_clahe)

# # new_layer = viewer.add_image(norm_N4)
            
            # ONLY RUN N4 on the autofluorescent
            if stripefilt and (ch == 1 or ch_auto == 'ch0'):

                im = np.asarray(im, dtype=np.uint16)
   
                tiff.imwrite(sav_dir + filename +'_level_' + level + '_ch' + str(ch) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution) + '_PRE_stripefilt.tif',  
                              np.expand_dims(np.expand_dims(np.asarray(im, dtype=np.uint16), axis=0), axis=2),
                              imagej=True, resolution=(1/(XY_res * down_XY), 1/(XY_res * down_XY)),
                              metadata={'spacing':Z_res * down_Z, 'unit': 'um', 'axes': 'TZCYX'})
                
                macro = """
                //@ String filepath
                //@ String outpath
                open(filepath);
                //run("Stripes Filter", "filter=Wavelet-FFT direction=Both types=Symlet wavelet=Sym5 border=[Symmetrical mirroring] image=don't negative entire decomposition=0:8 damping=4 large=100000000 small=1 tolerance=1 half=5 offset=1");
                //run("Stripes Filter", "filter=Wavelet-FFT direction=Both types=Symlet wavelet=Sym5 border=[Symmetrical mirroring] image=don't negative entire decomposition=0:3 damping=1.5 large=100000000 small=1 tolerance=1 half=5 offset=1");
                run("Stripes Filter", "filter=Wavelet-FFT direction=Both types=Symlet wavelet=Sym5 border=[Symmetrical mirroring] image=don't negative entire decomposition=0:5 damping=2 large=100000000 small=1 tolerance=1 half=5 offset=1");
                
                
                
                saveAs("Tiff", outpath);
                run("Close All");
                /// add close all later
                

                """
                args = {
                    'filepath': sav_dir + filename +'_level_' + level + '_ch' + str(ch) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution)  + '_PRE_stripefilt.tif',
                    'outpath': sav_dir + filename +'_level_' + level + '_ch' + str(ch) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution)  + '_STRIPE_FILT.tif'

                }
                #ij.py.run_macro(macro, args)
                ij.py.run_script('ijm', macro, args)
                
                im = tiff.imread(sav_dir + filename +'_level_' + level + '_ch' + str(ch) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution) + '_STRIPE_FILT.tif')
                
                ## MASK OUT after stripe filter to remove weird banding artifacts in background
                if mask:
                    im[im_bw == 0] = 0        
                    
                    ### then add back in the normal looking even background!!!
                    im_tmp[im_bw == 1] = 0
                    
                    im = im + im_tmp
                    
                    
                
                
            if n4_bool and (ch == 1 or ch_auto == 'ch0'):
                print('N4_filtering')
                

                # test2, field = N4_correction(im, mask_im=im_bw, shrinkFactor=8, numberFittingLevels=4)
                tiff.imwrite(sav_dir + filename +'_level_' + level + '_ch' + str(ch) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution) + '_PRE_n4.tif',  
                              np.expand_dims(np.expand_dims(np.asarray(im, dtype=np.uint16), axis=0), axis=2),
                              imagej=True, resolution=(1/(XY_res * down_XY), 1/(XY_res * down_XY)),
                              metadata={'spacing':Z_res * down_Z, 'unit': 'um', 'axes': 'TZCYX'})

                
                im, field = N4_correction(im, mask_im=im_bw, shrinkFactor=4, numberFittingLevels=4)
                # test3, field = N4_correction(im, mask_im=im_bw, shrinkFactor=1, numberFittingLevels=1)
                # import napari
                
                # viewer = napari.Viewer()
                # new_layer = viewer.add_image(test1)
                # new_layer = viewer.add_image(test2)
                # new_layer = viewer.add_image(test3)
                # new_layer = viewer.add_image(im)


                tiff.imwrite(sav_dir + filename +'_level_' + level + '_ch' + str(ch) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution) + '_POST_n4.tif',  
                              np.expand_dims(np.expand_dims(np.asarray(im, dtype=np.uint16), axis=0), axis=2),
                              imagej=True, resolution=(1/(XY_res * down_XY), 1/(XY_res * down_XY)),
                              metadata={'spacing':Z_res * down_Z, 'unit': 'um', 'axes': 'TZCYX'})
  

                                           
                




            
            if add_myelin and (ch == 1 or ch_auto == 'ch0'):
                
                im = im + masked_filt
                
                

                
            # else:
            #     dset = np.asarray(im, dtype=np.uint16)
            
            dset = np.asarray(im, dtype=np.uint16)

            ### save unpadded
            tiff.imwrite(sav_dir + filename +'_level_' + level + '_ch' + str(ch) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution)  + '_NOPAD.tif',  
                         np.expand_dims(np.expand_dims(np.asarray(dset, dtype=np.uint16), axis=0), axis=2),
                         imagej=True, resolution=(1/(XY_res * down_XY), 1/(XY_res * down_XY)),
                         metadata={'spacing':Z_res * down_Z, 'unit': 'um', 'axes': 'TZCYX'})
            
            
                        
            
            """ EXPAND IMAGE TO AVOID EDGE ARTIFIACTS? """
            if pad:
                #padded = np.pad(dset, ((z_before, z_after), (x_before, x_after), (x_before, x_after)), constant_values=(constant_val,))

                constant_val = np.median(dset[:, 0, 0])
                padded = np.pad(dset, ((z_before, z_after), (x_before, x_after), (x_before, x_after)), constant_values=(constant_val,))
                dset = padded    
             
                ### save image
                tiff.imwrite(sav_dir + filename +'_level_' + level + '_ch' + str(ch) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution) + '_PAD.tif',  
                             np.expand_dims(np.expand_dims(np.asarray(dset, dtype=np.uint16), axis=0), axis=2),
                             imagej=True, resolution=(1/(XY_res * down_XY), 1/(XY_res * down_XY)),
                             metadata={'spacing':Z_res * down_Z, 'unit': 'um', 'axes': 'TZCYX'})
            
                print('saved channel ' + str(ch))
            
        
        
        #%%
        """ Add option to register with olfactory and cerebellum masked out??? """
        
        
    
        
    #%% If divide by myelin
    ch_reg = ch_auto
    if divide_myelin:            
        
        #thresh = 500 ### GOOD FOR YOUNGER BRAINS --- older use 700
        print('EXTRACTING MYELIN FOR DIVISION with THRESH: ')
        print(thresh_div)
        
        
        myelin_im = tiff.imread(sav_dir + filename +'_level_' + level + '_ch' + str(0) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution) + '_NOPAD.tif')
        autofluor = tiff.imread(sav_dir + filename +'_level_' + level + '_' + ch_auto + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution) + '_NOPAD.tif')
        
        div_im, extracted_myelin = div_by_myel(myelin_im, autofluor=autofluor, thresh=thresh_div)
                
        
        ch_tmp = ch + 1  ### make 3rd channel for registration!
        ch_reg = 'ch' + str(ch_tmp)
        ### save unpadded
        tiff.imwrite(sav_dir + filename +'_level_' + level + '_ch' + str(ch_tmp) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution) + '_NOPAD.tif',  
                     np.expand_dims(np.expand_dims(np.asarray(div_im, dtype=np.uint16), axis=0), axis=2),
                     imagej=True, resolution=(1/(XY_res * down_XY), 1/(XY_res * down_XY)),
                     metadata={'spacing':Z_res * down_Z, 'unit': 'um', 'axes': 'TZCYX'})

        ### save extracted myelin image
        plot_max(extracted_myelin)
        tiff.imwrite(sav_dir + filename +'_level_' + level + '_ch' + str(ch_tmp) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution) + '_EXTRACTED_MYELIN.tif',  
                     np.expand_dims(np.expand_dims(np.asarray(extracted_myelin, dtype=np.uint16), axis=0), axis=2),
                     imagej=True, resolution=(1/(XY_res * down_XY), 1/(XY_res * down_XY)),
                     metadata={'spacing':Z_res * down_Z, 'unit': 'um', 'axes': 'TZCYX'})
                      
                      
                            
        if pad:
            constant_val = np.median(autofluor[:, 0, 0])
            padded = np.pad(div_im, ((z_before, z_after), (x_before, x_after), (x_before, x_after)), constant_values=(constant_val,))
            dset = padded    
            ### save image
            tiff.imwrite(sav_dir + filename +'_level_' + level + '_ch' + str(ch_tmp) + '_n4_down' + str(n4_bool) + '_resolution_' + str(resolution) + '_PAD.tif',  
                         np.expand_dims(np.expand_dims(np.asarray(dset, dtype=np.uint16), axis=0), axis=2),
                         imagej=True, resolution=(1/(XY_res * down_XY), 1/(XY_res * down_XY)),
                         metadata={'spacing':Z_res * down_Z, 'unit': 'um', 'axes': 'TZCYX'})
        
            print('saved channel ' + str(ch_tmp))        


    
    
        
    #%% Run BrainReg
    print('Starting BrainReg')
    
    if n4_bool:
        # images = glob.glob(os.path.join(sav_dir,'*down1_PAD.tif'))    # get all tiffs saved from above

        images = glob.glob(os.path.join(sav_dir,'*resolution_' + str(resolution) + '_PAD.tif'))    # get all tiffs saved from above
        

        
    else:
        # images = glob.glob(os.path.join(sav_dir,'*down0_PAD.tif'))

        images = glob.glob(os.path.join(sav_dir, '*resolution_' + str(resolution) + '_PAD.tif'))
     
        
    add_chs = []
    reg_name = []
    for name in images: 
        if ch_reg in name: reg_name = name
        else: add_chs.append(name)
    
    
    print(reg_name)
        
    # XY_res_scaled = XY_res * down_XY
    # Z_res_scaled = Z_res * down_Z
    
    
    ### SCALING
    #if n4_bool:
    if scale:
        XY_res_scaled = (XY_res/XY_scale) * down_XY
        Z_res_scaled = (Z_res/Z_scale) * down_Z

    else:
      XY_res_scaled = XY_res * down_XY
      Z_res_scaled = Z_res * down_Z     
      
    
    
    whole_brain_voxel_sizes = (str(Z_res_scaled), str(XY_res_scaled), str(XY_res_scaled))
    # grid_spacing = '-15'     ### worse with -5 than -10
    
    
    grid_spacing = '-10'     ### worse with -5 than -10
    
    #grid_spacing = '-2'    
             
    bending_energy = '0.9'  # was 0.95, testing with expanded brains was best with 0.5?
    
    ### helpful guide for orientation https://brainglobe.info/documentation/general/image-definition.html
    
    #atlas = 'princeton_mouse_20um'
    #atlas = 'perens_lsfm_mouse_20um'
    
    #atlas = '633_perens_ANTS_20um'
    
    #atlas = '633_princeton_mouse_20um'
    #atlas = '633_princeton_UPDATED_mouse_20um'
    
    #atlas = '633_princeton_tmp_REG_mouse_20um'
    #atlas = '633_perens_lsfm_mouse_20um'
    #atlas = '633_perens_STRIPED_lsfm_mouse_20um'
    #atlas = '633_perens_M115_lsfm_mouse_20um'
    #atlas = '633_perens_MYELIN_lsfm_mouse_20um'
    #atlas = '633_perens_lsfm_tmp_REG_mouse_20um'
    
    #atlas = '633_perens_lsfm_mouse_20um_RIGHT'
    
    
    #atlas = '488_perens_lsfm_mouse_20um'
    
    # atlas = 'allen_mouse_10um'
    #atlas = 'allen_mouse_25um'
    
    if divide_myelin:
        
        
        atlas = 'allen_mouse_20um'        
        if resolution == 10:
            atlas = 'allen_mouse_10um'
    else:
        atlas = 'allen_mouse_20um_CUBIC_FULLBRAIN'
       
    
    # atlas = 'allen_mouse_20um' 
    
    
    orientation = 'ial'
    
    num_cpus = '8'
    
    smooth_gauss = '0'   ### default -1 *** which is in voxels!!!
    use_steps = 'default'
    
    
    #whole_brain_data_dir = sav_dir + filename + atlas + '_N4_corr_NO_BACK' + bending_energy + '_grid_' + grid_spacing + '_gauss_' + smooth_gauss + '_PADDED'
    
    #whole_brain_data_dir = sav_dir + filename + atlas + '_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_' + bending_energy + '_n4_' + str(n4_bool) + '_grid_' + grid_spacing + '_gauss_' + smooth_gauss  + '_use_steps_' + str(use_steps) + '_PADDED_' + str(x_after)

    whole_brain_data_dir = sav_dir + filename + atlas + '_CLEANED_N4_corr_SCALE_AHE_STRIPEFILT_NOGAUSS_n4_DIVIDE_MYELIN_' + bending_energy + '_n4_' + str(n4_bool) + '_grid_' + grid_spacing + '_gauss_' + smooth_gauss  + '_use_steps_' + str(use_steps) + '_PADDED_' + str(x_after)
   
    

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
        "--debug",
        "--grid-spacing", grid_spacing, ### negative value in voxels, positive in mm, smaller means more local deforms  default: -10
    
        "--bending-energy-weight", bending_energy, ### between 0 and 1 (default: 0.95)
    
        "--smoothing-sigma-reference", '0', ### adds gaussian to reference image # default 1 is bad
        "--smoothing-sigma-floating", smooth_gauss,  ### adds gaussian to moving image # default 1 is bad
        

        # "--affine-n-steps", use_steps,
        # "--affine-use-n-steps", use_steps,
        
        # "--freeform-n-steps", use_steps,
        # "--freeform-use-n-steps", use_steps,
        
        
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
    


            
            