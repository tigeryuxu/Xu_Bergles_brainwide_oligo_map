#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:24:27 2020

@author: user
"""


from skimage import measure
import numpy as np
from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.UNet_functions_PYTORCH import *
from functional.matlab_crop_function import *

import time
import progressbar

""" Get an image from a dataframe """
def gen_im_frame_from_array(tracked_cells_df, frame_num, input_im, color=0):
    truth_im = np.zeros(np.shape(input_im))
    
    for idx_truth in np.where(tracked_cells_df.FRAME == frame_num)[0]:
        if not color:
            truth_im[tracked_cells_df.iloc[idx_truth].coords[:, 0], tracked_cells_df.iloc[idx_truth].coords[:, 1], tracked_cells_df.iloc[idx_truth].coords[:, 2]] = tracked_cells_df.iloc[idx_truth].SERIES 
        else:
            rgb = tracked_cells_df.iloc[idx_truth].COLOR
            if rgb == 'GREEN':
                truth_im[tracked_cells_df.iloc[idx_truth].coords[:, 0], tracked_cells_df.iloc[idx_truth].coords[:, 1], tracked_cells_df.iloc[idx_truth].coords[:, 2]] = 1
            elif rgb == 'RED':
                truth_im[tracked_cells_df.iloc[idx_truth].coords[:, 0], tracked_cells_df.iloc[idx_truth].coords[:, 1], tracked_cells_df.iloc[idx_truth].coords[:, 2]] = 2
            elif rgb == 'YELLOW':
                truth_im[tracked_cells_df.iloc[idx_truth].coords[:, 0], tracked_cells_df.iloc[idx_truth].coords[:, 1], tracked_cells_df.iloc[idx_truth].coords[:, 2]] = 3

    
    return truth_im
     

""" Get new cells and terminated cells ONLY from a dataframe """
def gen_im_new_term_from_array(tracked_cells_df, frame_num, input_im, new=0):
    truth_im = np.zeros(np.shape(input_im))
    
    for idx_truth in np.where(tracked_cells_df.FRAME == frame_num)[0]:
        
        ### to get only terminated cells
        if not new and frame_num < np.max(tracked_cells_df.FRAME):  ### EXCLUDING LAST FRAME
            if tracked_cells_df.iloc[idx_truth + 1].FRAME <= frame_num:   ### MEANS TERMINATED
                truth_im[tracked_cells_df.iloc[idx_truth].coords[:, 0], tracked_cells_df.iloc[idx_truth].coords[:, 1], tracked_cells_df.iloc[idx_truth].coords[:, 2]] = tracked_cells_df.iloc[idx_truth].SERIES 
        
        ### get only NEWLY FORMED cells, excluding if on 1st frame
        elif frame_num > 0:
            
            cell = tracked_cells_df.iloc[idx_truth]
            series = cell.SERIES
            
            prev_cell = np.where((tracked_cells_df.FRAME == frame_num - 1) & (tracked_cells_df.SERIES == series))[0]
            

            
            if len(prev_cell) == 0:   ### MEANS NEWLY FORMED
                print(tracked_cells_df.iloc[idx_truth].SERIES )
                truth_im[tracked_cells_df.iloc[idx_truth].coords[:, 0], tracked_cells_df.iloc[idx_truth].coords[:, 1], tracked_cells_df.iloc[idx_truth].coords[:, 2]] = tracked_cells_df.iloc[idx_truth].SERIES 
                
    return truth_im
    


### to load from truth_array
def gen_im_frame_from_TRUTH_array(truth_array, frame_num, input_im, lowest_z_depth, height_tmp, width_tmp, depth_tmp, scale=0):
    truth_im = np.zeros(np.shape(input_im))
    
    """ SCALE EVERYTHING IF NEEDED """
    # if scale:
    #      im_x_size = width_tmp
    #      im_y_size = height_tmp
    #      im_z_size = depth_tmp
         
    #      x_scale = 1/0.8302662
    #      y_scale = 1/0.8302662
    #      z_scale = 1/3
    #      truth_array.X = truth_array.X * x_scale;
    #      truth_array.Y = truth_array.Y * y_scale;
    #      # Normalize to first val 0 indexing
    #      middle_val = im_x_size / 2;
    #      truth_array.X = round(truth_array.X + middle_val);
    #      middle_val = im_y_size / 2;
    #      truth_array.Y = round(truth_array.Y + middle_val);
         
    #      #%% Scale Z
    #      truth_array.Z = truth_array.Z * z_scale;
    #      middle_val = im_z_size / 2;
    #      truth_array.Z = round(truth_array.Z + middle_val);
         
    #      #%% Tiger - add row of index
    #      #indices = 1:length(frame);    

    for idx_truth in np.where(truth_array.FRAME == frame_num)[0]:
         if idx_truth == 0:
              continue
         if truth_array.iloc[idx_truth].Z < lowest_z_depth and truth_array.iloc[idx_truth].Y < height_tmp and truth_array.iloc[idx_truth].X < width_tmp:
              truth_im[int(truth_array.iloc[idx_truth].Y), int(truth_array.iloc[idx_truth].X), int(truth_array.iloc[idx_truth].Z)] = truth_array.iloc[idx_truth].SERIES
              
              
    return truth_im

     
     
     
     
   
""" generate truth from csv """
def gen_truth_from_csv(frame_num, input_path, filename, input_im, lowest_z_depth, height_tmp, width_tmp, depth_tmp, scale=0, swap=1):
    truth_array = pd.read_csv(input_path + filename, sep=',')
    #truth_array = truth_array.sort_values(by=['FRAME'])
    
    
    """ SCALE EVERYTHING IF NEEDED """
    if scale:
         im_x_size = height_tmp
         im_y_size = width_tmp
         im_z_size = depth_tmp
         
         x_scale = 1/0.8302662
         y_scale = 1/0.8302662
         z_scale = 1/3
         truth_array.X = truth_array.X * x_scale;
         truth_array.Y = truth_array.Y * y_scale;
         # Normalize to first val 0 indexing
         middle_val = im_x_size / 2;
         truth_array.X = round(truth_array.X + middle_val);
         middle_val = im_y_size / 2;
         truth_array.Y = round(truth_array.Y + middle_val);
         
         #%% Scale Z
         truth_array.Z = truth_array.Z * z_scale;
         middle_val = im_z_size / 2;
         truth_array.Z = round(truth_array.Z + middle_val);
         
         #%% Tiger - add row of index
         #indices = 1:length(frame);    
    
    
    
    truth_im = np.zeros(np.shape(input_im))
    
    for idx_truth in np.where(truth_array.FRAME == frame_num)[0]:
         
         if swap:
              if truth_array.iloc[idx_truth].Z < lowest_z_depth and truth_array.iloc[idx_truth].Y < width_tmp and truth_array.iloc[idx_truth].X < height_tmp:
                   truth_im[int(truth_array.iloc[idx_truth].Y), int(truth_array.iloc[idx_truth].X), int(truth_array.iloc[idx_truth].Z)] = truth_array.iloc[idx_truth].SERIES 
                   
         # else:   ### for MATLAB full auto output
         #       if truth_array.Z[idx_truth] < lowest_z_depth and truth_array.X[idx_truth] < height_tmp and truth_array.Y[idx_truth] < width_tmp:
         #            truth_im[int(truth_array.X[idx_truth]) + 1, int(truth_array.Y[idx_truth]) + 1, int(truth_array.Z[idx_truth]) + 1] = truth_array.SERIES[idx_truth]               
              

    return truth_im, truth_array




""" Crop and prep input to prepare for CNN 

"""
def prep_input_for_CNN(cell, input_im, next_input, cur_seg, next_seg, mean_arr, std_arr, x, y, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp, next_bool=1, retry=0):

    ### (1) crop the current frame (all segmentations)
    crop_cur_seg, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(cur_seg, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
   
    ### (2) create the target seed for the current frame and crop it out
    blank_im = np.zeros(np.shape(input_im))
    blank_im[cell.coords[:, 0], cell.coords[:, 1], cell.coords[:, 2]] = 1                 
    crop_seed, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(blank_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
    
    ### (3) set the target seed to be 50, and the other segmentations in the frame to be 10
    crop_cur_seg[crop_cur_seg > 0] = 10
    crop_cur_seg[crop_seed > 0] = 50                 
       
    ### (4) crop the raw input data for the current and next segmentation
    crop_im, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(input_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
    crop_next_input, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(next_input, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                         
    
    ### (5) get segmentation for next frame
    crop_next_seg, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(next_seg, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      
    crop_next_seg_non_bin = np.copy(crop_next_seg)
    
    """ for testing doubles again at the end, eliminate all previously checked"""
    if retry:
        bw = crop_next_seg
        bw[bw > 0] = 1
        label = measure.label(bw)
        cc = measure.regionprops(label, intensity_image=crop_next_seg_non_bin)
        for obj in cc:
            coord = obj['coords']
            min_int = obj['min_intensity']
            if min_int == 250:  ### remove visited before cells
                bw[coord[:, 0], coord[:, 1], coord[:, 2]] = 0
                
        crop_next_seg = bw
        
        
    
    crop_next_seg[crop_next_seg > 0] = 10
  
    
    """ Get ready for inference """
    if next_bool:
        batch_x = np.zeros((4, ) + np.shape(crop_im))
        batch_x[0,...] = crop_im
        batch_x[1,...] = crop_cur_seg
        batch_x[2,...] = crop_next_input
        batch_x[3,...] = crop_next_seg
        batch_x = np.moveaxis(batch_x, -1, 1)
        batch_x = np.expand_dims(batch_x, axis=0)

    else:
        batch_x = np.zeros((3, ) + np.shape(crop_im))
        batch_x[0,...] = crop_im
        batch_x[1,...] = crop_cur_seg
        batch_x[2,...] = crop_next_input
        #batch_x[3,...] = crop_next_seg
        batch_x = np.moveaxis(batch_x, -1, 1)
        batch_x = np.expand_dims(batch_x, axis=0)

    
    ### NORMALIZE
    batch_x = normalize(batch_x, mean_arr, std_arr)

    return batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, crop_next_seg_non_bin, box_xyz, box_over



""" When there is more than one cell/object identified in next frame,
        need to associate the one that is BEST MATCHED with the original frame
"""
def select_one_from_excess(seg_train, crop_next_seg, crop_next_input):
    label = measure.label(seg_train)
    cc_seg_train = measure.regionprops(label, intensity_image=crop_next_input)
    if len(cc_seg_train) > 1:
         #doubles += 1
         #print('multi objects in seg')
         
         
         ### pick the ideal one ==> confidence??? distance??? and then set track color to 'YELLOW'
         """ Best one is one that takes up most of the area in the "next_seg" """
         #add = crop_next_seg + seg_train
         label = measure.label(crop_next_seg)
         cc_next = measure.regionprops(label)
         
         
         best = np.zeros(np.shape(crop_next_seg))
         
         all_ratios = []
         all_lens = []
         for multi_check in cc_seg_train:
              coords_m = multi_check['coords']
              #crop_next_seg[coords_m[:, 0], coords_m[:, 1], coords_m[:, 2]]
              #print('out')
              all_lens.append(len(coords_m))
              all_cur_r = [];
              for seg_check in cc_next:
                   coords_n = seg_check['coords']
                  
                   if np.any((coords_m[:, None] == coords_n).all(-1).any(-1)):   ### overlapped
                        ratio = len(coords_m)/len(coords_n)
                        all_cur_r.append(ratio)
                        
              if len(all_cur_r) > 0:
                   all_ratios.append(all_cur_r[np.argmin(np.abs(np.asarray(all_cur_r) - 1))])  ### OTHERWISE MIGHT MATCH EVEN MORE???
                             ### SHOULD ONLY APPEND RATIO OF LARGEST MATCHED
                        
         
               
         ### if there is ratio
         if len(all_ratios) > 0:
             best_coords = cc_seg_train[np.argmin(np.abs(np.asarray(all_ratios) - 1))]['coords']
             best[best_coords[:, 0], best_coords[:, 1], best_coords[:, 2]] = 1
             
         ### otherwise, if no cell matches between the 2 frames, pick the largest cell
         else:
             best_coords = cc_seg_train[np.argmax(all_lens)]['coords']
             best[best_coords[:, 0], best_coords[:, 1], best_coords[:, 2]] = 1

         seg_train = best
         label = measure.label(seg_train)
         cc_seg_train = measure.regionprops(label, intensity_image=crop_next_input)      
         
         
    return cc_seg_train, seg_train



""" When there is more than one cell/object identified in next frame,
        need to associate the one that is BEST MATCHED with the original frame
"""
def match_to_existing(seg_train, crop_next_seg):
    label = measure.label(seg_train)
    cc_seg_train = measure.regionprops(label)
         
    ### pick the ideal one ==> confidence??? distance??? and then set track color to 'YELLOW'
    """ Best one is one that takes up most of the area in the "next_seg" """
    #add = crop_next_seg + seg_train
    label = measure.label(crop_next_seg)
    cc_next = measure.regionprops(label)
    
    
    best = np.zeros(np.shape(crop_next_seg))
    overlapped = 0;
    for multi_check in cc_seg_train:
         coords_m = multi_check['coords']
         for seg_check in cc_next:
              coords_n = seg_check['coords']
             
              if np.any((coords_m[:, None] == coords_n).all(-1).any(-1)):   ### overlapped
                   best[coords_n[:, 0], coords_n[:, 1], coords_n[:, 2]] = 1
                   overlapped += 1
                   

    ### if something overlaps, use the overlap, otherwise don't                   
    if overlapped:
        seg_train = best
        label = measure.label(seg_train)
        cc_seg_train = measure.regionprops(label)      
         
    return cc_seg_train, seg_train, overlapped




""" Find cell number that matches in next_frame (i.e. where more than 1 cell 
         from cur_frame points to a cell in next frame)
"""
def sort_double_linked(tracked_cells_df, next_centroid, frame_num):
    ### pick the ideal one ==> confidence??? distance??? and then set track color to 'YELLOW'
    """ Find cell number that matches in next_frame (i.e. where more than 1 cell from cur_frame points to a cell in next frame)"""
    dup_series = []
    dup_series_index = []

    for row_tup in tracked_cells_df.loc[tracked_cells_df['FRAME'].isin([frame_num])].itertuples():
        
        cell = row_tup
        index = getattr(cell, 'Index')
        x =  getattr(cell, 'X'); y =  getattr(cell, 'Y'); z =  getattr(cell, 'Z');
        series = getattr(cell, 'SERIES')
        coords = getattr(cell, 'coords')
       
        
        if any((next_centroid == x).all() for x in coords):
              dup_series.append(series)
              
    
              
    return tracked_cells_df, dup_series







""" Parse the data and compare to ground truth for later """

def parse_truth(truth_cur_im, truth_array, truth_output_df, truth_next_im, seg_train, crop_next_seg, crop_seed, list_exclude, frame_num, x, y, z, crop_size, z_size, blobs, TP, FP, TN, FN, extras, height_tmp, width_tmp, depth_tmp):
   
    crop_truth_cur, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(truth_cur_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                         
    crop_truth_next, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(truth_next_im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                                                      

    """ DEBUG """               
    print('TPs = ' + str(TP) + '; FPs = ' + str(FP) + '; TNs = ' + str(TN) + '; FNs = ' + str(FN) + '; extras = ' + str(extras))
    
    seg_train = dilate_by_ball_to_binary(seg_train, radius = 3)  ### DILATE A BIT
    crop_next_seg = dilate_by_ball_to_binary(crop_next_seg, radius = 3)  ### DILATE A BIT
    crop_seed = dilate_by_ball_to_binary(crop_seed, radius = 3)
    
    
    """ REMOVE EVERYTHING IN CROP_NEXT_SEG THAT DOES NOT MATCH WITH SOMETHING CODY PUT UP, to prevent FPs of unknown checking"""
    # if nothing in the second frame
    value_cur_frame = np.unique(crop_truth_cur[crop_seed > 0])
    value_cur_frame = np.delete(value_cur_frame, np.where(value_cur_frame == 0)[0][0])  # DELETE zero
    
    values_next_frame = np.unique(crop_truth_next[crop_next_seg > 0])

    
    ### skip if no match on cur frame in truth
    if len(value_cur_frame) == 0:
         #not_registered += 1;  
         print('not_registered')
         return TP, FP, TN, FN, extras, blobs, truth_output_df, truth_array, list_exclude
    
    if not np.any(np.in1d(value_cur_frame, values_next_frame)):   ### if it does NOT exist on next frame               
         ### BUT IF EXISTS IN GENERAL ON 2nd frame, just not in the segmentation, then skip ==> is segmentation missed error
         values_next_frame_all = np.unique(crop_truth_next[crop_truth_next > 0])
         if np.any(np.in1d(value_cur_frame, values_next_frame_all)):
              #seg_error += 1;
              print('seg_error')
              list_exclude.append(value_cur_frame[0])
              return TP, FP, TN, FN, extras, blobs, truth_output_df, truth_array, list_exclude  # SKIP
         
         ### count blobs:
         if len(value_cur_frame) > 1:
              blobs += 1
              #blobs = 1
       
         ### AND if seg_train says it does NOT exist ==> then is TRUE NEGATIVE
         if np.count_nonzero(seg_train) == 0:       
              TN += 1     
         else:
              ### otherwise, all of the objects are False positives
              #FP += len(cc_seg_train)
              FP += 1
              
    else:
         
         ### (1) if seg_train is empty ==> then is a FALSE NEGATIVE
         if np.count_nonzero(seg_train) == 0:
              FN += 1
              
              """ Missing a lot of little tiny ones that are dim
                        *** maybe at the end add these back in??? by just finding nearest unassociated ones???
              """
         else:
              
            ### (2) find out if seg_train has identified point with same index as previous frame
            values_next_frame = np.unique(crop_truth_next[seg_train > 0])
            values_next_frame = np.delete(values_next_frame, np.where(values_next_frame == 0)[0][0])  # delete zeros
         
            if np.any(np.in1d(value_cur_frame, values_next_frame)):
                   TP += 1
                   values_next_frame = np.delete(values_next_frame, np.where(values_next_frame == values_next_frame)[0][0])
              
            """ if this is first time here, then also add the ones from initial index """
            if frame_num == 1:
                 row  = truth_array[(truth_array["SERIES"] == np.max(value_cur_frame)) & (truth_array["FRAME"] == 0)]
                 truth_output_df = truth_output_df.append(row)                                      
            row  = truth_array[(truth_array["SERIES"] == np.max(value_cur_frame)) & (truth_array["FRAME"] == frame_num)]
            truth_output_df = truth_output_df.append(row) 
            
            # but if have more false positives
            if len(values_next_frame) > 0:
              #FP += len(values_next_frame)
              extras += len(values_next_frame)
    #plt.close('all')  
    
    
    return TP, FP, TN, FN, extras, blobs, truth_output_df, truth_array, list_exclude


""" Associate remainder as newly segmented cells """
def associate_remainder_as_new(tracked_cells_df, next_seg, frame_num, lowest_z_depth, z_size, next_input, min_size=0, truth=0, truth_output_df=0, truth_next_im=0, truth_array=0):
    bw_next_seg = np.copy(next_seg)
    
    """ DON'T USE BINARY, b/c actually if touchs a larger new cell, might set it to be old cell!!!
    
            ... or maybe binary is better    
    """
    
    bw_next_seg[bw_next_seg > 0] = 1
    
    labelled = measure.label(bw_next_seg)
    next_cc = measure.regionprops(labelled, intensity_image=next_input)

    debug = 0
    if debug:
        debug_seg = np.copy(next_seg)
        debug_seg[debug_seg == 255] = 1 
        debug_seg[debug_seg == 250] = 2 
        
        debug_empty = np.zeros(np.shape(next_seg))
      
      
    ### add the cells from the first frame into "tracked_cells" matrix
    num_new = 0; num_new_truth = 0
    for idx, cell in enumerate(next_cc):
       coords = cell['coords']
       centroid = cell['centroid']
       intensity = cell['mean_intensity']
       
       
       if not np.any(next_seg[coords[:, 0], coords[:, 1], coords[:, 2]] == 250) and len(coords) > min_size:   ### 250 means already has been visited
            series = np.max(tracked_cells_df.SERIES) + 1   
            
            if debug:
                print(len(coords))
                
                # if len(coords) == 2944:
                #     break;
                    
                batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, crop_next_seg_non_bin, box_xyz, box_over = prep_input_for_CNN(cell, next_input, input_im, next_seg,
                                                                                                          cur_seg_LINKED, mean_arr, std_arr, centroid[0], centroid[1], centroid[2], crop_size, z_size,
                                                                                                          height_tmp, width_tmp, depth_tmp, next_bool=next_bool)
                plot_max(crop_im, ax=-1)
                plot_max(crop_cur_seg, ax=-1)
                
                debug_empty[coords[:, 0], coords[:, 1], coords[:, 2]] = debug_empty[coords[:, 0], coords[:, 1], coords[:, 2]] + 1
                plot_max(debug_empty, ax=-1)
                
                
                crop_debug, empty, empty, empty = crop_around_centroid_with_pads(debug_empty, int(coords[0][1]), int(coords[0][0]), int(coords[0][2]), crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)      
                
            
            """ SKIP IF HAVE TO MOVE THE CROPPING BOX IN THE BOTTOM Z-dimension """
            # if int(centroid[2]) + z_size/2 >= lowest_z_depth:
            #       continue                        
            
            row = {'SERIES': series, 'COLOR': 'BLANK', 'FRAME': frame_num, 'X': int(centroid[0]), 'Y':int(centroid[1]), 'Z': int(centroid[2]), 
                   'coords':coords, 'mean_intensity':intensity, 'visited': 0}
            tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)
                                  
            """ Add to TRUTH as well """
            if truth:
                value_next_frame = np.max(truth_next_im[coords[:, 0], coords[:, 1], coords[:, 2]])
                if value_next_frame > 0:
                     row  = truth_array[(truth_array["SERIES"] == np.max(value_next_frame)) & (truth_array["FRAME"] == frame_num)]
                     if truth:
                         truth_output_df = truth_output_df.append(row) 
                     print('value_next_frame')
                     num_new_truth += 1
                 
            num_new += 1
            
       else:
            if debug:
                debug_seg[coords[:, 0], coords[:, 1], coords[:, 2]] = 2
            


            
       #print('Checking cell: ' + str(idx) + ' of total: ' + str(len(next_cc)))
    

    print('num new cells: ' + str(num_new))
                    
    return tracked_cells_df, truth_output_df, truth_next_im, truth_array
   
    
    
    



""" Given an image with segmentations for the next frame + coordinates of center of current frame
        find closest cell to associate current cell to within min_dist of 20        

"""
def associate_to_closest(tracked_cells_df, cc, x, y, z, box_xyz, box_over, cur_idx, frame_num, width_tmp, height_tmp, depth_tmp, min_dist=20):

    all_dist = []
    for obj in cc:
        center = obj['centroid']
        
        center = np.asarray(center)
        center = scale_single_coord_to_full(center, box_xyz, box_over)
         
        dist = [x, y, z] - center
        dist = np.linalg.norm(dist)
        all_dist.append(dist)
         
   
    """ only keep if smallest is within 20 pixels """
    closest_obj = cc[np.argmin(all_dist)]
    closest_dist = np.min(all_dist)
   
    #print(closest_dist)
   
    next_coords = []; next_centroid = []; cell_next = []; next_intensity = [];
    if closest_dist <= min_dist:
        index_next = np.where((tracked_cells_df["SERIES"] == cur_idx) & (tracked_cells_df["FRAME"] == frame_num))[0]

        cell_next = tracked_cells_df.loc[tracked_cells_df['FRAME'].isin([frame_num]) & tracked_cells_df['SERIES'].isin([cur_idx])]
        index_next = cell_next.index
        
        if len(index_next) > 0:        
            cell_next = tracked_cells_df.loc[index_next[0]]
        
        next_intensity = np.asarray(closest_obj['mean_intensity'])
        
        next_coords = np.asarray(closest_obj['coords'])
        #seg_train = np.zeros(np.shape(seg_train))
        #seg_train[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 1
        
        next_coords = scale_coords_of_crop_to_full(next_coords, box_xyz, box_over)
   
   
        next_centroid = np.asarray(closest_obj['centroid'])
        next_centroid = scale_single_coord_to_full(next_centroid, box_xyz, box_over)
        
        """ Need to check to ensure the coords do not go over the image size limit because the CNN output is NOT subjected to the limits!!! """
        next_coords = check_limits([next_coords], width_tmp, height_tmp, depth_tmp)[0]
        next_centroid = check_limits_single([next_centroid], width_tmp, height_tmp, depth_tmp)[0]       
        
            
    return cell_next, next_coords, next_centroid, closest_dist, next_intensity



""" Predict next xyz given cells in current crop frame of arbitrary size       

"""
def predict_next_xyz(tracked_cells_df, x, y, z, crop_size, z_size, frame_num):

    cell_locs =  tracked_cells_df.index[tracked_cells_df['X'].isin(range(int(x-crop_size/2),int(x+crop_size/2))) 
                                        & tracked_cells_df['Y'].isin(range(int(y-crop_size/2),int(y+crop_size/2)))
                                        & tracked_cells_df['Z'].isin(range(int(z-z_size/2),int(z+z_size/2)))
                                        & tracked_cells_df['FRAME'].isin([frame_num - 1])]    
           
    tracked_locs_in_crop = []
                
    cur_centers = []; next_centers = []; num_tracked = 0
    for row_tup in tracked_cells_df.loc[cell_locs].itertuples():
        
        cell = row_tup
        series = getattr(cell, 'SERIES')
        index = getattr(cell, 'Index')
        x_c = getattr(cell, 'X'); y_c = getattr(cell, 'Y'); z_c = getattr(cell, 'Z');        
        
        next_cell_loc =  tracked_cells_df.index[tracked_cells_df['SERIES'].isin([series]) 
                                        & tracked_cells_df['FRAME'].isin([frame_num])]       
     
        if len(next_cell_loc) > 0:
            next_cell = tracked_cells_df.loc[next_cell_loc]
            index_next = next_cell.index
             
     
            cur_centroid = [x_c, y_c, z_c]
            next_centroid = [np.asarray(next_cell.X)[0], np.asarray(next_cell.Y)[0], np.asarray(next_cell.Z)[0]]
            
            cur_centers.append(cur_centroid)
            next_centers.append(next_centroid)
            
            num_tracked += 1
            tracked_locs_in_crop.append(index)
            
            
    if len(next_centers) > 0:
        all_vectors = (np.asarray(next_centers) - np.asarray(cur_centers))
        
        median_disp = [np.median(all_vectors[:, 0]), np.median(all_vectors[:, 1]), np.median(all_vectors[:, 2])]
   
        pred_x = int(x + median_disp[0])
        pred_y = int(y + median_disp[1])
        pred_z = int(z + median_disp[2])

    else:
        pred_x = x
        pred_y = y
        pred_z = z
        
        num_tracked = 0
    
    return pred_x, pred_y, pred_z, num_tracked, tracked_locs_in_crop




""" Help with changing the pointer of the old cell or adding a new cell"""
def change_pointer_or_add_cell(tracked_cells_df, next_seg, cell_next, cell, frame_num, next_coords, next_centroid, moved_old, new, next_intensity):

    ### change pointer of old cell
    if len(cell_next) > 0:
        old_coords = cell_next.coords
        next_seg[old_coords[:, 0], old_coords[:, 1], old_coords[:, 2]] = 255;   ### RESET NEXT_SEG 
        
        
        cell_next.coords = next_coords
        cell_next.X = next_centroid[0]
        cell_next.Y = next_centroid[1]
        cell_next.Z = next_centroid[2]
        
        next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;   ### RESET NEXT_SEG 
        
        moved_old += 1
        
        
    ### or add new cell
    else:
        row = {'SERIES': cell.SERIES, 'COLOR': 'GREEN', 'FRAME': frame_num, 'X': int(next_centroid[0]), 'Y':int(next_centroid[1]), 'Z': int(next_centroid[2]), 
               'coords':next_coords, 'mean_intensity': next_intensity, 'visited': 0}
        #tracked_cells_df = tracked_cells_df.append(row, ignore_index=True)     ### THIS WILL RE-ARRANGE ALL INDICES!!!
        tracked_cells_df.loc[np.max(np.asarray(tracked_cells_df.index)) + 1] = row


        """ Change next coord """
        next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 250;                      
        
        #print('new cell')
        new += 1
        
    return tracked_cells_df, next_seg, moved_old, new

                
                
""" Use predictions to cleanup whatever candidates you wish to try """
def clean_with_predictions(tracked_cells_df, candidate_series, next_seg, crop_size, z_size, frame_num, height_tmp, width_tmp, depth_tmp, input_im=0, next_input=0, cur_seg = 0, min_dist=12):
    
    debug = 0
    print('cleaning with predictions')
    #start = time.perf_counter()
    deleted = 0; term_count = 0; new = 0; not_changed = 0; moved_old = 0;
    not_assoc = 0;
    to_drop = [];
    recheck_series = []   ### TO BE RECHECKED LATER   
    
    for row_tup in tracked_cells_df.loc[tracked_cells_df['FRAME'].isin([frame_num - 1]) & tracked_cells_df['SERIES'].isin(candidate_series)].itertuples():
        
        cell = row_tup
        index = getattr(cell, 'Index')
        x =  getattr(cell, 'X'); y =  getattr(cell, 'Y'); z =  getattr(cell, 'Z');
        series = getattr(cell, 'SERIES')
        
        cur_cell_coords = getattr(cell, 'coords')
       
        # if series == 763:
        #     zzz
        #print(series)             
        
        ### DEBUG: when debugging get next cell too and plot it
        #im = np.zeros(np.shape(next_seg))
        cell_next = tracked_cells_df.loc[tracked_cells_df['FRAME'].isin([frame_num]) & tracked_cells_df['SERIES'].isin([series])]
        index_next = cell_next.index
        
        unassociated_bool = 0
        if len(index_next) > 0:
            cell_next = tracked_cells_df.loc[index_next[0]]
            x_n = cell_next.X; y_n = cell_next.Y; z_n = cell_next.Z;

        else:
            cell_next = []
            unassociated_bool = 1
            
        ### DEBUG:
        im = np.zeros(np.shape(next_seg))
        batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, crop_next_seg_non_bin, box_xyz, box_over = prep_input_for_CNN(cell, input_im, next_input, cur_seg,
                                                                                                  next_seg, 0, 0, x, y, z, crop_size, z_size,
                                                                                                  height_tmp, width_tmp, depth_tmp, next_bool=1)   
        if debug:
            plot_max(crop_im, ax=-1)
            plot_max(crop_cur_seg, ax=-1)
            plot_max(crop_next_input, ax=-1)
    
            crop_next_seg_non_bin[crop_next_seg_non_bin == 250] = 1
            crop_next_seg_non_bin[crop_next_seg_non_bin == 255] = 2
         
            plot_max(crop_next_seg_non_bin, ax=-1)
            
            crop_next_seg, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(next_seg, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)                    

        """ now associate cell to the object closest to the predicted location
                a few things can happen:
                        (1) no cell currently occupied, so just associate
                        (2) currently occupied, in which case, remove that association, and add that cell to the new list of cells to check
                        (3) no cell found nearby, in which case, set as terminated
        """            
        
        ### use predicted xyz only if num_tracked > 5:
        bw = crop_next_seg; bw[bw > 0] = 1
        label = measure.label(crop_next_seg)
        cc = measure.regionprops(label, intensity_image=crop_next_input)
        next_coords = []
        if len(cc) > 0:
        
            num_tracked = 0; scale = 0.5
            while num_tracked < 4 and scale <= 3:
                pred_x, pred_y, pred_z, num_tracked, tracked_locs_in_crop = predict_next_xyz(tracked_cells_df, x, y, z, crop_size + crop_size * scale, z_size + z_size * scale, frame_num)
                
                scale += 0.25 
                 
            ### Try to associate with nearest cell in crop_next_seg        
            if num_tracked >= 4:
                empty, next_coords, next_centroid, closest_dist, next_intensity = associate_to_closest(tracked_cells_df, cc, pred_x, pred_y, pred_z, box_xyz, box_over, series, 
                                                                                                 frame_num, width_tmp, height_tmp, depth_tmp, min_dist=min_dist)       
            
            
        """ If this cell was previously unassociated, make sure that the next cell is not exorbitantly large!!!
        
        
        """
        if unassociated_bool:
            if len(next_coords) > 0:
                #print('unassociated, check size'); print(len(cur_cell_coords)); print(len(next_coords));
                # #if len(cur_cell_coords) < 500 and len(next_coords) > 1000:
                if (len(cur_cell_coords) > 500 or len(next_coords)  > 800) and len(next_coords) > len(cur_cell_coords) * 2:
                    not_changed += 1
                    
                    # plot_max(crop_im, ax=-1)
                    # plot_max(crop_cur_seg, ax=-1)
                    # plot_max(crop_next_input, ax=-1)
            
                    # crop_next_seg_non_bin[crop_next_seg_non_bin == 250] = 1
                    # crop_next_seg_non_bin[crop_next_seg_non_bin == 255] = 2
                 
                    # plot_max(crop_next_seg_non_bin, ax=-1)                   
                    # im[pred_x, pred_y, pred_z] = 2
                    # im[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 3
                    # crop_seg_out, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)       
                    # plot_max(crop_seg_out, ax=-1)                    
                    
                    
                    #print('skip since size difference is too big to be certain')
                    continue;
                    #next_coords = []
                    
                    #a = 1
        

        """ Change next coord only if something close was found
        """
        term_bool = 0
        if len(next_coords) > 0:   ### only add if not empty
            ### DEBUG:
            if debug:
                #im[x_n, y_n, z_n] = 1
                im[pred_x, pred_y, pred_z] = 2
                im[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 3
                crop_seg_out, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)       
                plot_max(crop_seg_out, ax=-1)


            """ CASE #1: if next_seg does NOT contain 250, then just associate to current cell"""
            if not np.any(next_seg[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] == 250):
                
                """ change the pointer of the old cell OR ADD CELL  
                            ***also must update "next_seg"
                    """
                tracked_cells_df, next_seg, moved_old, new = change_pointer_or_add_cell(tracked_cells_df, next_seg, cell_next, cell, frame_num, next_coords, next_centroid, moved_old, new, next_intensity)
                  
                #print('hello')
                #zzz

            else:
                """ CASE #2: otherwise, find which cell is currently matched to this cell, then check what the prediction is for that new cell
                    and find which is closer 
                    
                    """
                found = 0
                # while not found:
                #     empty, empty, empty, empty, tracked_locs_in_crop = predict_next_xyz(tracked_cells_df, x, y, z, crop_size + crop_size * scale, z_size + z_size * scale, frame_num)
                    
                #     scale += 0.25 
 
                for row_tup_check in tracked_cells_df.loc[tracked_locs_in_crop].itertuples():
                
                    cell_check_cur = row_tup_check    
                    series_check = getattr(cell_check_cur, 'SERIES')
                    #                    
                    next_cell_check = tracked_cells_df.loc[tracked_cells_df['FRAME'].isin([frame_num]) & tracked_cells_df['SERIES'].isin([series_check])]
                    index_next_c = next_cell_check.index
                                  
                    """ SKIP BECAUSE if own cell is the closest, then do nothing!"""
                    ### i.e. upon re-testing, will come here often
                    if series == series_check:
                        if len(next_cell_check.coords) > 0:
                            x_c = int(next_cell_check.X); y_c = int(next_cell_check.Y); z_c = int(next_cell_check.Z); 
                            if len(np.where((next_coords == (x_c, y_c, z_c)).all(axis=1))[0]) > 0:
                                print('matched self only')
                                
                        found = 1
                        break;
                    
                    
                    """ If matched a different cell """
                    if len(next_cell_check.coords) > 0:
                        
                       
                        
                        ### check if matched
                        x_c = int(next_cell_check.X); y_c = int(next_cell_check.Y); z_c = int(next_cell_check.Z); 
                        
                        
                        ### find if row matches row in next_coords
                        if len(np.where((next_coords == (x_c, y_c, z_c)).all(axis=1))[0]) > 0:
                            
                            
                            #print(series_check)
                            #print('matched')
                            found += 1
                            # if series == 930:
                            #     zzz
                            
                            
                            ### HACK: rare case where cells on first frame will double hit 2 cells very near to each other. In this case, just assess one of them for now???
                            if found == 2:
                                break;
                            

                                                        
                            """ Predict where this cell is going and see which is closer """                                    
                            x_c_cur = getattr(cell_check_cur, 'X'); y_c_cur = getattr(cell_check_cur, 'Y'); z_c_cur = getattr(cell_check_cur, 'Z'); 
                
                            ### keep looping until have sufficient neighbor landmarks
                            num_tracked = 0; scale = 0.5
                            while num_tracked < 4 and scale <= 2:
                                pred_x_c, pred_y_c, pred_z_c, num_tracked, empty = predict_next_xyz(tracked_cells_df, x_c_cur, y_c_cur, z_c_cur, crop_size + crop_size * scale, z_size + z_size * scale, frame_num)
                                
                                scale += 0.25                            
                            
                            
                            """ Find distances to the next cell using predictions from 2 cells on current frame """
                            dist_to_check = np.linalg.norm(np.asarray([pred_x_c, pred_y_c, pred_z_c]) - next_centroid)
                            dist_to_new = np.linalg.norm(np.asarray([pred_x, pred_y, pred_z]) - next_centroid)
                            
                            ### if the new cell is closer, than DELETE the pointer from the "check" cell
                            if dist_to_new <= dist_to_check:
                                #print('delete cell')
                                deleted += 1
                                to_drop.append(index_next_c[0])   ### drop NEXT FRAMES cell that was conflicting
                                
                                
                                # print(to_drop)
                                # if index_next_c == 4180:
                                #     zzz
                                #next_seg[next_cell_check.coords[:, 0], next_cell_check.coords[:, 1], next_cell_check.coords[:, 2]] = 255;   ### RESET NEXT_SEG 
                                
                                ### and append the cell who's pointer is removed, so we can check it again later
                                recheck_series.append(series_check)
                                
                                
                                #print(series_check)

                                """ change the pointer of the old cell OR ADD CELL  
                                            ***also must update "next_seg"
                                    """
                                
                                tracked_cells_df, next_seg, moved_old, new = change_pointer_or_add_cell(tracked_cells_df, next_seg, cell_next, cell, frame_num, next_coords, next_centroid, moved_old, new, next_intensity)
                                
                                #print('yee')
                                # print(moved_old); print(new)
                                #zzz
                                
                            else:
                                ### otherwise, leave current cell as empty
                                #print('not linked')
                                term_bool = 1;
                                term_count += 1
                                
                        
                                ### DELETE THE COORDS OF THE "NEW" CELL
                                # if len(cell_next) > 0:
                                #     old_coords = cell_next.coords
                                #     next_seg[old_coords[:, 0], old_coords[:, 1], old_coords[:, 2]] = 255;   ### RESET NEXT_SEG 
                                
                                if debug:
                                    im[next_coords[:, 0], next_coords[:, 1], next_coords[:, 2]] = 0
                                    crop_seg_out, box_xyz, box_over, boundaries_crop  = crop_around_centroid_with_pads(im, y, x, z, crop_size/2, z_size, height_tmp, width_tmp, depth_tmp)       
                                    plot_max(crop_seg_out, ax=-1)

                                
                     
                      
                    ### MIGHT BE BUG???
                    ### for now, if didn't match, then just add as new cell
                    
                    """ Means that the ONLY cell that matched was itself!!!
                            so it stayed it's own cell
                    
                    """
                    
                if found == 0:
                    not_assoc += 1
                    print('NOT FOUND')
                    #print(not_assoc)
                    """ change the pointer of the old cell OR ADD CELL  
                                ***also must update "next_seg"
                    """                    
                    tracked_cells_df, next_seg, moved_old, new = change_pointer_or_add_cell(tracked_cells_df, next_seg, cell_next, cell, frame_num, next_coords, next_centroid, moved_old, new, next_intensity)
                    #print('huh')
                    #zzz
                    
        
 
        """   CASE #3: none matched, set as eliminated and remove cell_next """
                                
        if len(next_coords) == 0 or term_bool:
           
            if len(index_next) > 0:
                to_drop.append(index_next[0])
                deleted += 1
                
                old_coords = cell_next.coords
                next_seg[old_coords[:, 0], old_coords[:, 1], old_coords[:, 2]] = 255;   ### RESET NEXT_SEG 

                
            
            else:
                not_changed += 1
           
                     

    """ drop everything that has .coords = [] """
    tracked_cells_df = tracked_cells_df.drop(to_drop)
    print('new associations: ' + str(new) + '\ndeleted_old_tracks: ' + str(deleted) + '\nterminated: ' + str(term_count) + '\nnot changed: ' + str(not_changed) + '\nmoved: ' + str(moved_old))
    
    #stop = time.perf_counter(); diff = stop - start; print(diff);
    
    
    return tracked_cells_df, recheck_series, next_seg






### check new cells

""" PLOT new cells """
# all_cur_frame = np.where(tracked_cells_df["FRAME"] == frame_num)[0]
# cur_series = tracked_cells_df.iloc[all_cur_frame].SERIES
# new_series = []
# for cur in cur_series:
#     index_cur = np.where((tracked_cells_df["SERIES"] == cur) & (tracked_cells_df["FRAME"] == frame_num - 1))[0]
#     index_next = np.where((tracked_cells_df["SERIES"] == cur) & (tracked_cells_df["FRAME"] == frame_num))[0]
     
#     """ if next frame is empty, then terminated """
#     if len(index_cur) == 0:
#         new_series.append(cur)
        
        
# print('cleaning with predictions')
# for series in new_series:

#     index = np.where((tracked_cells_df["SERIES"] == series) & (tracked_cells_df["FRAME"] == frame_num))[0]
#     cell = tracked_cells_df.iloc[index[0]]
#     x = cell.X; y = cell.Y; z = cell.Z;
#     pred_x, pred_y, pred_z, num_tracked = predict_next_xyz(tracked_cells_df, x, y, z, crop_size, z_size, frame_num)
#     print(len(cell.coords))

#     ### DEBUG: when debugging get next cell too and plot it
#     #im = np.zeros(np.shape(next_seg))
#     index_next = np.where((tracked_cells_df["SERIES"] == series) & (tracked_cells_df["FRAME"] == frame_num - 1))[0]
#     if len(index_next) > 0:
#         cell_next = tracked_cells_df.iloc[index_next[0]]
#         x_n = cell_next.X; y_n = cell_next.Y; z_n = cell_next.Z;

#     else:
#         cell_next = []
        
#     im = np.zeros(np.shape(next_seg))

#     batch_x, crop_im, crop_cur_seg, crop_seed, crop_next_input, crop_next_seg, crop_next_seg_non_bin, box_xyz, box_over = prep_input_for_CNN(cell, next_input, input_im, next_seg,
#                                                                                               cur_seg, 0, 0, x, y, z, crop_size, z_size,
#                                                                                               height_tmp, width_tmp, depth_tmp, next_bool=next_bool)   

#     plot_max(crop_im, ax=-1)
#     plot_max(crop_cur_seg, ax=-1)
#     plot_max(crop_next_input, ax=-1)

#     crop_next_seg_non_bin[crop_next_seg_non_bin == 250] = 1
#     crop_next_seg_non_bin[crop_next_seg_non_bin == 255] = 2
 
#     plot_max(crop_next_seg_non_bin, ax=-1)
    
    
    
    
    
    
    
    


""" Given dataframe, find predicted distances of ALL

        can be used to find possible errors
        
        ***also used to plot show distribution of how accurate Cody's traces are to predicting each other!!!
"""



def check_predicted_distances(tracked_cells_df, frame_num, crop_size, z_size, dist_error_thresh = 10):
    print('checking distances')
    all_dist = []; check_series = []; dist_check = []; num_checked = 0;
    
    # unique_series =  np.unique(tracked_cells_df.SERIES)    
    # idx_series = tracked_cells_df.index[tracked_cells_df['SERIES'].isin(unique_series) & ]
            
    for row_tup in progressbar.progressbar(tracked_cells_df.loc[tracked_cells_df['FRAME'].isin([frame_num - 1])].itertuples(), max_value=len(tracked_cells_df.loc[tracked_cells_df['FRAME'].isin([frame_num - 1])]), redirect_stdout=True):
                
            num_checked += 1
            
            #print(num_checked)
            
            cell = row_tup
             
            ### go to unvisited cells
            x = getattr(cell, 'X'); y = getattr(cell, 'Y'); z = getattr(cell, 'Z');   
            series = getattr(cell, 'SERIES')
            
            #print(num_checked)
            
            ### keep looping until have sufficient neighbor landmarks
            num_tracked = 0; scale = 0.25
            while num_tracked < 4 and scale <= 2:
                pred_x, pred_y, pred_z, num_tracked, tracked_locs_in_crop = predict_next_xyz(tracked_cells_df, x, y, z, crop_size + crop_size * scale, z_size + z_size * scale, frame_num)
                
                scale += 0.25
            
            if num_tracked >= 4:
                index_next = tracked_cells_df.index[tracked_cells_df['SERIES'].isin([series]) & tracked_cells_df['FRAME'].isin([frame_num])]
                
                
                if len(index_next) > 0:
                    cell_next = tracked_cells_df.loc[index_next[0]]    
    
                    x_n = cell_next.X; y_n = cell_next.Y; z_n = cell_next.Z; 
                    
                    dist = np.linalg.norm(np.asarray([x_n, y_n, z_n]) - np.asarray([pred_x, pred_y, pred_z]))
                    all_dist.append(dist)
                    #print(dist)
                    
                    if dist > dist_error_thresh:
                        check_series.append(series)
                        dist_check.append(dist)                   
                        
    # len(np.where(np.asarray(all_dist) > 4)[0])
    # len(all_dist) 
    
    return tracked_cells_df, all_dist, dist_check, check_series





""" Directly compare tracking using .csv files """
def load_and_compare_csvs_to_truth(input_path, filename, examples, lowest_z_depth, truth_array, truth_name, truth_path, input_im, height_tmp, width_tmp, depth_total, scale):



    MATLAB_auto_array = pd.read_csv(input_path + filename, sep=',')
    
    
    all_cells_MATLAB = np.unique(MATLAB_auto_array.SERIES)
    all_cells_TRUTH = np.unique(truth_array.SERIES)


    all_lengths_MATLAB = []
    truth_lengths = []    
    MATLAB_lengths = []
    all_cell_nums = []
    for frame_num in range(len(examples)):
         print('Starting inference on volume: ' + str(frame_num) + ' of total: ' + str(len(examples)))
       
         seg_name = examples[frame_num]['seg']  
         seg = open_image_sequence_to_3D(seg_name, width_max='default', height_max='default', depth='default')
         seg = seg[0:lowest_z_depth, ...]
         seg = np.moveaxis(seg, 0, -1)       
         
         
    
         truth_next_im, truth_array  = gen_truth_from_csv(frame_num=frame_num, input_path=truth_path, filename=truth_name, 
                                   input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_total, scale=scale)
                          
         
         MATLAB_next_im, MATLAB_auto_array  = gen_truth_from_csv(frame_num=frame_num, input_path=input_path, filename=filename, 
                                   input_im=input_im, lowest_z_depth=lowest_z_depth, height_tmp=height_tmp, width_tmp=width_tmp,
                                   depth_tmp=depth_total, scale=0, swap=1)
         
         
         """ region props on seg ==> then loop through each individual cell to find out which numbers are matched
         
              then delete those numbers from the "all_cells_MATLAB" and "all_cells_TRUTH" matrices
              
              while finding out the lengths
         """
         label_seg = measure.label(seg)
         cc_seg = measure.regionprops(label_seg)
         
         for cell in cc_seg:
              coords = cell['coords']
              
              
              if np.any(truth_next_im[coords[:, 0], coords[:, 1], coords[:, 2]] > 0) and np.any(MATLAB_next_im[coords[:, 0], coords[:, 1], coords[:, 2]] > 0):
                  
                   num_truth = np.unique(truth_next_im[coords[:, 0], coords[:, 1], coords[:, 2]])
                   if len(np.intersect1d(all_cells_TRUTH, num_truth)) > 0:
                        num_new_truth = np.max(np.intersect1d(all_cells_TRUTH, num_truth)) ### only keep cells that haven't been tracked before                   
                        track_length_TRUTH = len(truth_array.loc[truth_array["SERIES"].isin([num_new_truth])])
                        
                        
                  
                        
                   """ get cell from MATLAB """
                   num_MATLAB = np.unique(MATLAB_next_im[coords[:, 0], coords[:, 1], coords[:, 2]])   
                   #print(num_MATLAB)
                   if len(np.intersect1d(all_cells_MATLAB, num_MATLAB)) > 0:
                       
                        num_new_MATLAB = np.max(np.intersect1d(all_cells_MATLAB, num_MATLAB)) ### only keep cells that haven't been tracked before
                        track_length_MATLAB = len(MATLAB_auto_array.loc[MATLAB_auto_array["SERIES"].isin([num_new_MATLAB])])
                        
                        
                      
                   
                   if len(np.intersect1d(all_cells_TRUTH, num_truth)) > 0 and len(np.intersect1d(all_cells_MATLAB, num_MATLAB)) > 0:
                   
                       all_lengths_MATLAB.append(track_length_TRUTH - track_length_MATLAB)
                       truth_lengths.append(track_length_TRUTH)
                       MATLAB_lengths.append(track_length_MATLAB)   
                        
                       all_cell_nums.append(num_new_truth)
                        
                       
                       all_cells_TRUTH = all_cells_TRUTH[all_cells_TRUTH != num_new_truth]   # remove this cell so cant be retracked
                        
                       all_cells_MATLAB = all_cells_MATLAB[all_cells_MATLAB != num_new_MATLAB]   # remove this cell so cant be retracked
               
               
    return all_lengths_MATLAB






