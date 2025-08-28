import numpy as np
import pandas as pd
import torch

import utils.exp_utils as utils
#import utils.model_utils as mutils

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')   

from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
from os import listdir
from os.path import isfile, join
import glob, os

from tifffile import *
import tkinter
from tkinter import filedialog
    
from skimage.measure import label, regionprops, regionprops_table

from inference_utils import *

from scipy.stats import norm

from functional.matlab_crop_function import *
from functional.tree_functions import *  

from predictor import apply_wbc_to_patient
import time
from multiprocessing.pool import ThreadPool
from tqdm import tqdm



# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(device)


def expand_add_stragglers(to_assign, clean_labels):
    
    to_assign = np.asarray(to_assign, np.int32)  ### required for measure.regionprops
    cc_ass = measure.regionprops(to_assign)
    for ass in cc_ass:
        
        coords = ass['coords']


        match = to_assign[coords[:, 0], coords[:, 1], coords[:, 2]]
        vals = np.unique(match)
        

        exp = expand_coord_to_neighborhood(coords, lower=1, upper=1 + 1)   ### always need +1 because of python indexing
        exp = np.vstack(exp)
        
        
        # make sure it doesnt go out of bounds
        exp[:, 0][exp[:, 0] >= clean_labels.shape[0]] = clean_labels.shape[0] - 1
        exp[:, 0][exp[:, 0] < 0] = 0
        
        
        exp[:, 1][exp[:, 1] >= clean_labels.shape[1]] = clean_labels.shape[1] - 1
        exp[:, 1][exp[:, 1] < 0] = 0


        exp[:, 2][exp[:, 2] >= clean_labels.shape[2]] = clean_labels.shape[2] - 1
        exp[:, 2][exp[:, 2] < 0] = 0
        
        
        
        

        values = clean_labels[exp[:, 0], exp[:, 1], exp[:, 2]]
        
        vals, counts = np.unique(values, return_counts=True)
        
        if np.max(vals) > 0:  ### if there is something other than background matched
            
            ### Assign to the nearest object with the MOST matches
            ass_val = np.argmax(counts[1:])   ### skip 0
            ass_val = vals[1:][ass_val]
            
            
            clean_labels[coords[:, 0], coords[:, 1], coords[:, 2]] = ass_val   ### Give these pixels the value of the associated object
            
        
            clean_labels = np.asarray(clean_labels, np.int32)
    return clean_labels
        



""" ***update this so it prints to a log file instead of to the terminal all this garbage"""

def post_process_async(cf, all_patches, im_size, filename, sav_dir, overlap_pxy, overlap_pz, patch_size, patch_depth, file_num, focal_cube, s_c=0, debug=0):
    print('run async for: ' + str(file_num))
    
    
    #im_size = np.shape(input_im); 
    width = im_size[1];  height = im_size[2]; depth_im = im_size[0];
    
    #filename = input_name.split('/')[-1].split('.')[0:-1]
    #filename = '.'.join(filename)

    # input_im = np.expand_dims(input_im, axis=0)
    # input_im = np.expand_dims(input_im, axis=2)
    # tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_input_im.tif', np.asarray(input_im, dtype=np.uint16),
    #                       imagej=True, #resolution=(1/XY_res, 1/XY_res),
    #                       metadata={'spacing':1, 'unit': 'um', 'axes': 'TZCYX'})  
                                 

    # segmentation = np.asarray(segmentation, np.uint8)
    # tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_segmentation_overlap3.tif', segmentation)
    
    ### READ SEGMENTATION FROM SAVED FILE
    segmentation = tiff.imread(sav_dir + filename + '_' + str(int(file_num)) +'_segmentation_overlap3.tif')

    #if debug:
        #colored_im = np.asarray(colored_im, np.int32)
        #tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_colored_im.tif', colored_im)
    
    
    ### Cleanup
    #colored_im = []; input_im = []; new_dim_im = []

    #%% For bounding box approach - SPEED UP
    print('wbc clustering for ' + str(len(all_patches)) + ' number of patches')
    tic = time.perf_counter()
    
    
    max_val = np.max(segmentation)
    intersect_levels = [[] for _ in range(max_val + 1)]
    pool_for_wbc = []
    exclude_edge = []
    less_overlap = []
    
    for patch in all_patches:
        
        xyz = patch['xyz']
        results = patch['results_dict']
        boxes = results['boxes'][0]
        
        #scaled_boxes = np.copy(results)
        for idb, box in enumerate(boxes):
            
            
            c = box['box_coords']

            box_centers = []                        
            
            ### Ensure it doesn't go out of bounds
            x_val = round((c[0] + c[2]) / 2)
            if x_val >= patch_size:
                x_val = patch_size - 1
            box_centers.append(x_val)   
            
            
            y_val = round((c[1] + c[3]) / 2)
            if y_val >= patch_size:
                y_val = patch_size - 1
            box_centers.append(y_val)   

            z_val = round((c[4] + c[5]) / 2)
            if z_val >= patch_depth:
                z_val = patch_depth - 1
            box_centers.append(z_val)  
  
            c[0] = c[0] + xyz[0]
            c[1] = c[1] + xyz[1]
            c[2] = c[2] + xyz[0]
            c[3] = c[3] + xyz[1]
            c[4] = c[4] + xyz[2]
            c[5] = c[5] + xyz[2]
            
            
 
            coords = np.copy(box['mask_coords']) 
            coords[:, 0] = coords[:, 0] + xyz[0]
            coords[:, 1] = coords[:, 1] + xyz[1]                             
            coords[:, 2] = coords[:, 2] + xyz[2]
            
            
            box['box_coords'] = c
            box['box_n_overlaps'] = 1
            box['patch_id'] = '0_0'
            box['mask_coords'] = coords
            
            
            ### Filter to improve speed - ones with high overlap (on overlapped regions), and ones with low (near middle of FOV)
            val = np.max(segmentation[coords[:, 2], coords[:, 0], coords[:, 1]])
            
            if val > 2:
                intersect_levels[0].append(box)
            
            else:
                intersect_levels[1].append(box)

            ### Add exclusion by focal_cube
            # box_centers = np.asarray(np.round(box_centers), dtype=int)
            # val = focal_cube[box_centers[0], box_centers[1], box_centers[2]]
            ### Uncomment below if want to use focal_cube exclusion
            # if val == 0:
            
            #     pool_for_wbc.append(box)
            
            # else:
            #     exclude_edge.append(box)   ### not needed if have overlap == 50%
            
    #tic = time.perf_counter()    
    all_boxes = []
    tic = time.perf_counter()
    for boxes_level in intersect_levels:  
        #print(len(boxes_level))
        # print('apply WBC')
        regress_flag = False
        n_ens = 1
        
        
        
        wbc_input = [regress_flag, [boxes_level], 'dummy_pid', cf.class_dict, cf.clustering_iou, n_ens]
        
        out = apply_wbc_to_patient(wbc_input)[0]   ### SLOW
        
        #all_boxes.append(out[0])
        
        all_boxes = all_boxes + out[0] ### concatenate lists
        
    
    toc = time.perf_counter()
    print(f"WBC in {toc - tic:0.4f} seconds")
    
    out = [all_boxes]
    
 
    #%% Get masks directly without any other postprocessing from maskrcnn outputs -- will have overlaps!!!
    # box_masks = []
    # for box in out[0]:
    #     box_masks.append(box['mask_coords'])

    # box_masks = np.asarray(box_masks[0])
    
    # tmp = np.zeros(np.shape(segmentation))
    # tmp_overlap = np.zeros(np.shape(segmentation))
    # for m_id, mask in enumerate(box_masks):
    #     tmp[mask[:, 2], mask[:, 0], mask[:, 1]] = m_id + 1
    #     tmp_overlap[mask[:, 2], mask[:, 0], mask[:, 1]]  = tmp_overlap[mask[:, 2], mask[:, 0], mask[:, 1]] + 1
    
    # tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'tmp.tif', np.asarray(tmp, dtype=np.int32))        
    # tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'tmp_overlap.tif', np.asarray(tmp_overlap, dtype=np.int32))        
    


    
    #%% For KNN based assignment of voxels to split boxes
    
    print('Splitting boxes with KNN')
    #patch_depth = patch_im.shape[0]
    #patch_size = patch_im.shape[1]

    box_vert = []
    for box in out[0]:
        box_vert.append(box['box_coords'])

    box_vert = np.asarray(np.round(np.vstack(box_vert)), dtype=int)
    
    
    seg_overall = segmentation
    
    
    df_cleaned = split_boxes_by_Voronoi3D(box_vert, vol_shape = seg_overall.shape)    ### SLOW
    merged_coords = df_cleaned['bbox_coords'].values
    
    
    ### Then APPLY these boxes to mask out the objects in the main segmentation!!!
    new_labels = np.zeros(np.shape(seg_overall))
    #overlap = np.zeros(np.shape(seg_overall))
    all_lens = []
    for box_id, box_coords in enumerate(merged_coords):
    
        if len(box_coords) == 0:
            continue
        all_lens.append(len(box_coords))
        

        bc = box_coords
        """ HACK: --- DOUBLE CHECK ON THIS SUBTRACTION HERE """
        
        #bc = bc - 1 ### cuz indices dont start from 0 from polygon function?
        bc = np.asarray(bc, dtype=int)                    
        new_labels[bc[:, 0], bc[:, 1], bc[:, 2]] = box_id + 1   
        #overlap[bc[:, 0], bc[:, 1], bc[:, 2]] = overlap[bc[:, 0], bc[:, 1], bc[:, 2]] + 1
        

    new_labels = np.asarray(new_labels, np.int32)


    
    # tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_clipped_boxes.tif', np.asarray(new_labels, np.uint32))  

    new_labels[seg_overall == 0] = 0   ### This is way simpler and faster than old method of looping through each detection
    #plot_max(new_labels)
    
    
    # tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_clipped_seg.tif', np.asarray(new_labels, np.uint32))  

    
    new_labels = np.asarray(new_labels, np.int32)



    toc = time.perf_counter()
    print(f"Step 1 in {toc - tic:0.4f} seconds")

    #%% CLEANUP
    """ CLEANUP 
    
            (1) Find objects that are not connected at all (spurious assignments) and only keep the larger object
            
            (2) The rest of the spurious assignments can be assigned to nearest object by dilating the spurious assignments
            
            (3) Same goes for leftover hanging bits of segmentation that were not assigned due to bounding box issues
    
    
    
            optional: do this earlier --> but delete any bounding boxes with a centroid that does NOT contain a segmented object?

    """
    
    
    print('Step 1: Cleaning up spurious assignments')
    tic = time.perf_counter()

              
    to_assign = np.zeros(np.shape(new_labels))
    cc = measure.regionprops(new_labels)
    im_shape = new_labels.shape

    obj_num = 1
    
    for id_o, obj in enumerate(cc):

        ### scale all the coords down first so it's plotted into a super small area!
        coords = obj['coords']
        diff = np.max(coords, axis=0) - np.min(coords, axis=0)
        
        tmp = np.zeros(diff + 1)
        scaled = coords - np.min(coords, axis=0)
        
        
        tmp[scaled[:, 0], scaled[:, 1], scaled[:, 2]] = 1

        ### If center is blank, also skip?
        # if tmp[center[0], center[1], center[2]] == 0:
            
        #     #clean_labels[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
        #     to_assign[coords[:, 0], coords[:, 1], coords[:, 2]] = obj_num
        #     print('empty center')
        #     obj_num += 1
        #     continue

            
        bw_lab = measure.label(tmp, connectivity=1)
        
        
        if np.max(bw_lab) > 1:
            check_cc = measure.regionprops(bw_lab)

            #print(str(id_o))
            all_lens = []
            all_coords = []
            for check in check_cc:
                all_lens.append(len(check['coords']))
                all_coords.append(check['coords'])
            
            #all_coords = np.asarray(all_coords)  ### here
            
            min_thresh = 30
            if np.max(all_lens) > min_thresh: ### keep the main object if it's large enough else delete everything by making it so it will 
                                              ### be re-assigned to actual nearest neighbor in the "to_assign" array below
            
                amax = np.argmax(all_lens)
                
                ### delete all objects that are NOT the largest conncected component
                ind = np.delete(np.arange(len(all_lens)), amax)
                to_ass = [all_coords[i] for i in ind]
                
            else:
                to_ass = all_coords

            ### Have to loop through coord by coord to make sure they remain separate
            for coord_ass in to_ass:
                
                ### scale coordinates back up
                coord_ass = coord_ass + np.min(coords, axis=0)
                
                to_assign[coord_ass[:, 0], coord_ass[:, 1], coord_ass[:, 2]] = obj_num
                
                obj_num += 1


    #to_assign = np.asarray(to_assign, np.int32)  ### required for measure.regionprops
    if debug:
       
        tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_to_assign_FOCAL.tif', to_assign)
                    
    
    
    ### Expand each to_assign to become a neighborhood!  ### OR JUST DILATE THE WHOLE IMAGE?
    clean_labels = new_labels
    clean_labels[to_assign > 0] = 0

    clean_labels = expand_add_stragglers(to_assign, clean_labels)
    if debug:
        tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_ass_step1.tif', clean_labels)                
    
    
    toc = time.perf_counter()
    print(f"Step 1 in {toc - tic:0.4f} seconds")
    
    #%% ## Expand each leftover segmentation piece to be a part of the neighborhood!
    # print('Step 2: Cleaning up neighborhoods')
    
    # bw_seg = segmentation
    # bw_seg[bw_seg > 0] = 1
    # bw_seg[clean_labels > 0] = 0
    
    
    # stragglers = measure.label(bw_seg)
    # clean_labels = expand_add_stragglers(stragglers, clean_labels)
    # ### Cleanup
    # bw_seg = []; stragglers = []
    # if debug:
    #     tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_ass_step2.tif', clean_labels)                      
    
    
    #%% ## Also clean up small objects and add them to nearest object that is large - ### SLOW!!!
    
    #tic = time.perf_counter()
    print('Step 3: Cleaning up adjacent objects')
    min_size = 80
    all_obj = measure.regionprops(clean_labels)
    small = np.zeros(np.shape(clean_labels))
    
    print(len(all_obj))
    
    #counter = 1
    for o_id, obj in enumerate(all_obj):
        c = obj['coords']
        obj_label = obj['label']
        
        if len(c) < min_size:
            small[c[:, 0], c[:, 1], c[:, 2]] = obj_label
            #counter += 1
 
        
    #toc = time.perf_counter()
    #print(f"clean up adjacent objects in {toc - tic:0.4f} seconds")                
         
    
    #small = np.asarray(small, np.int32)
    clean_labels[small > 0] = 0   ### must remember to mask out all the small areas otherwise will get reassociated back with the small area!
    clean_labels = expand_add_stragglers(small, clean_labels)

    ### Add back in all the small objects that were NOT near enough to touch anything else
    small[clean_labels > 0] = 0
    clean_labels[small > 0] = small[small > 0]
    

    
    if debug:
        tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_ass_step3_FOCAL.tif', clean_labels)                      



    #%% ## Also go through Z-slices and remove any super thin sections in XY? Like < 10 pixels
    
    
    print('Step 4: Cleaning up z-stragglers')
    count = 0
    for zid, zslice in enumerate(clean_labels):
        cc = measure.regionprops(zslice)
        for obj in cc:
            coords = obj['coords']
            if len(coords) < 10:
                clean_labels[zid, coords[:, 0], coords[:, 1]] = 0
                count += 1
                
    if debug:
        tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_ass_step4_FOCAL.tif', clean_labels)           
    

    ### Also remove super large objects?
    
    
    
    #%% Pad entire image and shift by 1 to the right and 1 down
    # print('Step 5: shifting segmentation by +1')
    # shift_im = np.zeros([depth_im + 1, width + 1, height + 1])
    # shift_im[1:, 1:, 1:] = clean_labels
    
    # shifted = shift_im[:-1, :-1, :-1]
    
    # shifted = np.asarray(shifted, np.int32)
    # tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_shifted.tif', shifted)    


    ### REMOVED SHIFTING AFTER NORMALIZATION
    print('NO SHIFTING AFTER NORMALIZATION')
    clean_labels = np.asarray(clean_labels, np.int32)
    tiff.imwrite(sav_dir + filename + '_' + str(int(file_num)) +'_NO_SHIFT.tif', clean_labels)    

    shifted = clean_labels


    ### Cleanup
    new_labels = []; clean_labels = []; segmentation = []; small = []; split_seg = []; shift_im = []
    tmp = []; to_assign = []; seg_overall = []
    
    toc = time.perf_counter()
    print(f"Total post-processing steps 1 - 5 in {toc - tic:0.4f} seconds")                
            

    #%% """ Also save list of coords of where the cells are located so that can easily access later (or plot without making a whole image!) """
    print('\nsaving coords')
    tic = time.perf_counter()
    
    #from skimage import measure
    #labels = measure.label(segmentation)
    #blobs_labels = measure.label(blobs, background=0)
    cc = measure.regionprops(shifted, cache=False)
    
    #cc_table = measure.regionprops_table(shifted, properties=('centroids', 'coords'))

    
    ########################## TOO SLOW TO USE APPEND TO ADD EACH ROW!!!
    ######################### MUCH FASTER TO JUST MAKE A DICTIONARY FIRST, AND THEN CONVERT TO DATAFRAME AND CONCAT
    #tic = time.perf_counter()
    d = {}
    
    #d = {'offset': s_c, 'block_num': file_num}'
    
    #all_c = []
    #all_coords = []
    #all_sc = []
    
    # for i in range(len(cd)):
        
    #     c = cd[i]['centroid']
    #     print(i)
    #     cd[i] = []
    
    
    for i_list, cell in enumerate(cc):
        
        #tic = time.perf_counter()
        center = cell['centroid']
        center = [round(center[0]), round(center[1]), round(center[2])]
        
        coords = cell['coords']
        coords_raw = np.copy(coords)


        coords[:, 0] = coords[:, 0] + s_c[2]
        coords[:, 1] = coords[:, 1] + s_c[1]
        coords[:, 2] = coords[:, 2] + s_c[0]
        
        coords = np.roll(coords, -1)  ### so that EVERYTHING is in XYZ format
        #print(i_list)
        
        
        ### check if anything is non-sequential
        # l = coords_raw[:, 1]
        # if not sorted(np.unique(l))== list(range(min(l), max(l)+1)):
        #     print('NOT SEQUENTIAL')
        #     print(cell['label'])
            # zzz

        
        ### BE CAREFUL - every new cell[parameter] requires heavy computation time from regionprops
        d[i_list] = {'xyz_offset': s_c, 'block_num': file_num, 
                'Z': center[0], 'X': center[1], 'Y': center[2],
                'Z_scaled': center[0] + s_c[2], 'X_scaled': center[1] + s_c[1], 'Y_scaled': center[2] + s_c[0],
                #'equiv_diam': cell['equivalent_diameter'], 
                #'vol': cell['area'], 
                'overlap_pxy':overlap_pxy, 'overlap_pz':overlap_pz,
                'coords_raw':coords_raw, 
                'coords_scaled':coords}
        
        cc[i_list] = []   ### need to manually garbage collect to prevent regionprops from caching and using up 100% RAM
        
        #print(i_list)
        
        
    # def cell_list(cell): 
    #     #tic = time.perf_counter()
    #     center = cell['centroid']
    #     center = [round(center[0]), round(center[1]), round(center[2])]
        
        
        
        
    #     coords = cell['coords']
    #     coords_raw = np.copy(coords)


    #     coords[:, 0] = coords[:, 0] + s_c[2]
    #     coords[:, 1] = coords[:, 1] + s_c[1]
    #     coords[:, 2] = coords[:, 2] + s_c[0]
        
    #     d = {#'offset': s_c, 'block_num': file_num, 
    #             'Z': center[0], 'X': center[1], 'Y': center[2],
    #             'Z_scaled': center[0] + s_c[2], 'X_scaled': center[1] + s_c[1], 'Y_scaled': center[2] + s_c[0],
    #             'equiv_diam': cell['equivalent_diameter'], 'vol': cell['area'], 'coords_raw':coords_raw, 'coords_scaled':coords}
    #     print(center)
        
        
    #     return d
        
    
        
    # pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    # result = pool.map(cell_list, cc)
    # pool.shutdown(wait=True)

    toc = time.perf_counter()
    print(f"Extracting coords in {toc - tic:0.4f} seconds")     
        
        
       
    df = pd.DataFrame.from_dict(d, "index")
    #coords_df = pd.concat([coords_df, df])           
                            
    
    df.to_pickle(sav_dir + filename + '_' + str(int(file_num)) + '_df.pkl')
    
    #a = pd.read_pickle(sav_dir + filename + '_' + str(int(file_num)) + '_df.pkl')
    
    
    #toc = time.perf_counter()
    
    #np.save(sav_dir + filename + '_' + str(int(file_num)) + '_numpy_arr', df)
    #print(f"Save coords in {toc - tic:0.4f} seconds \n\n")
    
    #print('Saved asynchronously')
                        
    #a = np.load(sav_dir + filename + '_' + str(int(file_num)) + '_numpy_arr.npy')


    #toc = time.perf_counter()
    #print(f"Saving coords in {toc - tic:0.4f} seconds")                
    print('DONE')    
    
    ### Cleanup
    shifted = [];

    
    
    return 1




#%%
    
if __name__=="__main__":
    class Args():
        def __init__(self):

            self.dataset_name = "datasets/OL_data"

            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/94) newest_CLEANED_shrunk_det_thresh_02_min_conf_01/'
            self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/96) new_FOV_data_det_thresh_09_check_300/'
            self.server_env = False


    args = Args()
    data_loader = utils.import_module('dl', os.path.join(args.dataset_name, "data_loader.py"))

    config_file = utils.import_module('cf', os.path.join(args.exp_dir, "configs.py"))
    cf = config_file.Configs()
    cf.exp_dir = args.exp_dir
    cf.test_dir = cf.exp_dir

    cf.fold = 0
    if cf.dim == 2:
        cf.merge_2D_to_3D_preds = True
        if cf.merge_2D_to_3D_preds:
            cf.dim==3
    else:
        cf.merge_2D_to_3D_preds = False
        
    cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(cf.fold))
    anal_dir = os.path.join(cf.exp_dir, "inference_analysis")

    logger = utils.get_logger(cf.exp_dir)
    model = utils.import_module('model', os.path.join(cf.exp_dir, "model.py"))
    torch.backends.cudnn.benchmark = cf.dim == 3
    

    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*_best_params.pth'))
    
    model_selector = utils.ModelSelector(cf, logger)

    starting_epoch = 1

    last_check = 1

    onlyfiles_check.sort(key = natsort_key1)
    weight_path = onlyfiles_check[-1]   ### ONLY SOME CHECKPOINTS WORK FOR SOME REASON???
    net = model.net(cf, logger).cuda(device)


    # load already trained model weights
    with torch.no_grad():
        pass
        
        if last_check:
            optimizer = torch.optim.AdamW(utils.parse_params_for_optim(net, weight_decay=cf.weight_decay,
                                                                   exclude_from_wd=cf.exclude_from_wd,
                                                                   ), 
                                      lr=cf.learning_rate[0])        
            checkpoint_path = os.path.join(cf.fold_dir, "last_state.pth")
            starting_epoch, net, optimizer, model_selector = \
                utils.load_checkpoint(checkpoint_path, net, optimizer, model_selector)
                
                
            net.eval()
            net.cuda(device)


        else:
    
            net.load_state_dict(torch.load(weight_path))
            net.eval()
            net = net.cuda(device)
        
        
        
        
    # generate a batch from test set and show results
    if not os.path.isdir(anal_dir):
        os.mkdir(anal_dir)


    """ Select multiple folders for analysis AND creates new subfolder for results output """
    root = tkinter.Tk()
    # get input folders
    another_folder = 'y';
    list_folder = []
    input_path = "./"
    
  
        

    list_folder = ['/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Test_RegRCNN/',
                   '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Test_RegRCNN_fortraining/']


    """ Loop through all the folders and do the analysis!!!"""
    for input_path in list_folder:
        foldername = input_path.split('/')[-2]
        sav_dir = input_path + '/' + foldername + '_output_PYTORCH_96_last300_skimage_COLORED_step3_thresh09_NOfoc_fixed_neighborhoods_ASYNC_speedupminor'
    
        """ For testing ILASTIK images """
        images = glob.glob(os.path.join(input_path,'*.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
        images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
        examples = [dict(input=i,truth=i.replace('.tif','_truth.tif'), mask=i.replace('.tif','_MASK.tif')) for i in images]


        try:
            # Create target Directory
            os.mkdir(sav_dir)
            print("Directory " , sav_dir ,  " Created ") 
        except FileExistsError:
            print("Directory " , sav_dir ,  " already exists")
            
        sav_dir = sav_dir + '/'
        

        input_im_stack = [];
        for file_num in range(len(examples)):
             
             with torch.set_grad_enabled(False):  # saves GPU RAM            
                input_name = examples[file_num]['input']            
                input_im = tiff.imread(input_name)
                
       
                """ Analyze each block with offset in all directions """ 
                print('Starting inference on volume: ' + str(file_num) + ' of total: ' + str(len(examples)))
                
                ### Define patch sizes
                patch_size=128; patch_depth=16
                
                ### Define overlap and focal cube to remove cells that fall within this edge
                overlap_pxy = 14; overlap_pz = 3
                step_xy = patch_size - overlap_pxy * 2
                step_z = patch_depth - overlap_pz * 2
                
                focal_cube = np.ones([patch_depth, patch_size, patch_size])
                focal_cube[overlap_pz:-overlap_pz, overlap_pxy:-overlap_pxy, overlap_pxy:-overlap_pxy] = 0                
                focal_cube = np.moveaxis(focal_cube, 0, -1)

                thresh = 0.9
                
                #thresh = 0.9
                cf.merge_3D_iou = thresh
                
                im_size = np.shape(input_im); width = im_size[1];  height = im_size[2]; depth_im = im_size[0];

                
                ### Add padding to input_im
                factorx = 0
                while factorx < width/step_xy:   factorx+=1
                end_padx = (factorx * step_xy) - width
                
                factory = 0
                while factory < height/step_xy:   factory+=1
                end_pady = (factory * step_xy) - height
                
                factorz = 0
                while factorz < depth_im/step_z:   factorz+=1
                end_padz = (factorz * step_z) - depth_im     
                
                print('Total num of patches: ' + str(factorx * factory * factorz))
                new_dim_im = np.zeros([overlap_pz*2 + depth_im + end_padz, overlap_pxy*2 + width + end_padx, overlap_pxy*2 + height + end_pady])
                new_dim_im[overlap_pz: overlap_pz + depth_im, overlap_pxy: overlap_pxy + width, overlap_pxy: overlap_pxy + height] = input_im
                input_im = new_dim_im
                
                im_size = np.shape(input_im); width = im_size[1];  height = im_size[2]; depth_im = im_size[0];                
                
                
                ### Define empty items
                box_coords_all = []; total_blocks = 0;
                segmentation = np.zeros([depth_im, width, height])
                colored_im = np.zeros([depth_im, width, height])
                
                split_seg = np.zeros([depth_im, width, height])
                all_xyz = [] 
                
                
                all_patches = []
                
                batch_size = 1
                batch_im = []
                batch_xyz = []
                
                
                debug = 0
                
                
                ### SET THE STEP SIZE TO BE HALF OF THE IMAGE SIZE
                #step_z = patch_depth/2
                #step_xy = patch_size/2
                
                ### add progress bar
                # estimate number of blocks
                pbar = tqdm(total=factorx * factory * factorz,
                            desc="Loading",
                            ncols = 75)
                
                
                
                
                #%% START MASK-RCNN analysis

                for z in range(0, depth_im + patch_depth, round(step_z)):
                    if z + patch_depth > depth_im:  continue

                    for x in range(0, width + patch_size, round(step_xy)):
                          if x + patch_size > width:  continue

                          for y in range(0, height + patch_size, round(step_xy)):
                               if y + patch_size > height: continue

                               #print([x, y, z])

                               all_xyz.append([x, y, z])


                               
                               quad_intensity = input_im[z:z + patch_depth,  x:x + patch_size, y:y + patch_size];  
                               quad_intensity = np.moveaxis(quad_intensity, 0, -1)
                               quad_intensity = np.asarray(np.expand_dims(np.expand_dims(quad_intensity, axis=0), axis=0), dtype=np.float16)
                               

                               if len(batch_im) > 0:
                                   batch_im = np.concatenate((batch_im, quad_intensity))
                               else:
                                   batch_im = quad_intensity
                               
                               
                               batch_xyz.append([x,y,z])
                               
                               #print(total_blocks)
                               total_blocks += 1  
                               
                               
                               if total_blocks % batch_size == 0:
      

                                   batch = {'data':batch_im, 'seg': np.zeros([batch_size, 1, patch_size, patch_size, patch_depth]), 
                                             'class_targets': np.asarray([]), 'bb_targets': np.asarray([]), 
                                             'roi_masks': np.zeros([batch_size, 1, 1, patch_size, patch_size, patch_depth]),
                                             'patient_bb_target': np.asarray([]), 'original_img_shape': quad_intensity.shape,
                                             'patient_class_targets': np.asarray([]), 'pid': ['0']}
                                   
                                   
                                   
                                   
                                   output = net.test_forward(batch) #seg preds are only seg_logits! need to take argmax.
                                   

                                   ### Add box patch factor
                                   new_box_list = []
                                   tmp_check = np.zeros(np.shape(quad_intensity[0][0]))
                                   for bid, box in enumerate(output['boxes'][0]):
                                        """ 
                                       
                                       
                                            Some bounding boxes have no associated segmentations!!! Skip these
                                       
                                       
                                       
                                        """
                                        if len(box['mask_coords']) == 0:
                                            continue
                                                                            
                                        ### Only add if above threshold
                                        if box['box_score'] > thresh:
                                            c = box['box_coords']

                                            box_centers = [(c[ii] + c[ii + 2]) / 2 for ii in range(2)]
                                            box_centers.append((c[4] + c[5]) / 2)

                                            # factor = np.mean([norm.pdf(bc, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 for bc, pc in \
                                            #                   zip(box_centers, np.array(quad_intensity[0][0].shape) / 2)]
                                            # slightly faster call
                                            pc =  np.array(quad_intensity[0][0].shape) / 2
                                            factor = np.mean([norm.pdf(box_centers, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 ])
                                                
                                                
                                            box['box_patch_center_factor'] = factor
                                            new_box_list.append(box)     
                                           
      
                                   
                                   output['boxes'] = [new_box_list]
                                   
                                   
                                   results_dict = output
                                   

    
                                   if 'seg_preds' in results_dict.keys():
                                        results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
                                       # results_dict['colored_boxes'] = np.expand_dims(results_dict['colored_boxes'][:, 1, :, :, :], axis=0)
                                        
                                   ### Add to segmentation
                                   seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1) 
                                   
                                   #color_im = np.moveaxis(results_dict['colored_boxes'], -1, 1) 
                                   segmentation[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] = segmentation[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] + seg_im[0, :, 0, :, :]
                                   #colored_im[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] = color_im[0, :, 0, :, :]
                                   
                                   for bs in range(batch_size):
    
                                       # box_df = results_dict['boxes'][bs]
                                       # box_vert = []
                                       # box_score = []
                                       # mask_coords = []
                                       # for box in box_df:
                                          
                                       #         box_vert.append(box['box_coords'])
                                       #         box_score.append(box['box_score'])
                                       #         mask_coords.append(box['mask_coords'])
                                               
                                       # if len(box_vert) == 0:
                                       #     continue
        
        
                                       patch_im = batch_im[bs][0]
                                       #patch_im = np.moveaxis(patch_im, -1, 0)   ### must be depth first
                                        
                                       # save memory by deleting seg_preds
                                       results_dict['seg_preds'] = []
                    
                                       all_patches.append({'results_dict': results_dict, 'total_blocks': bs % total_blocks + (total_blocks - batch_size), 
                                                                #'focal_cube':focal_cube, 'patch_im': patch_im, 'box_vert':box_vert, 'mask_coords':mask_coords
                                                               'xyz':batch_xyz[bs]})
          
                                       
                                   ### Reset batch
                                   batch_im = []
                                   batch_xyz = []
                                   
                                   patch = np.moveaxis(quad_intensity, -1, 1)
                                   save = np.concatenate((patch, seg_im), axis=2)
                                   
                                   pbar.update(1)
                                   
                pbar.close()

               




            




        #%% Try running asynchronously
        #zzz        
        
        
        ### Initiate poolThread
        poolThread = ThreadPool(processes=1)
        
        ### get NEXT tile asynchronously!!!

        poolThread.apply_async(post_process_async, (cf, input_im, segmentation, input_name, sav_dir, all_patches, 
                                                    patch_im, patch_size, patch_depth, file_num, focal_cube, debug)) 
        poolThread.close()
            
        
        
        
        zzz
        #poolThread.join()
            
                    
                
    #pool.join() ### called at VERY VERY END??? to make sure all tasks finish
    
# callback function
# def custom_callback(result):
# 	print(f'Got result: {result}')
    
# # issue a task asynchronously to the thread pool with a callback
# result = pool.apply_async(task, callback=custom_callback)