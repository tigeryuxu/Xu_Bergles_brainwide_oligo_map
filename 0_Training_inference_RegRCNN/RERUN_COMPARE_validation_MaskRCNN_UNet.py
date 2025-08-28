#!/usr/bin/env python
# Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" execution script. this where all routines come together and the only script you need to call.
    refer to parse args below to see options for execution.
"""




#%%  LOAD UNET

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# from UNet_pytorch import *
# from UNet_pytorch_online import *
# from PYTORCH_dataloader import *
# from UNet_functions_PYTORCH import *

import tifffile as tiff

# import plotting as plg

import matplotlib.pyplot as plt
import os
import warnings
import argparse
import time

import torch

import numpy as np
import tifffile as tiff
    

    
### TIGER ADDED:
torch.cuda.set_device(0)
torch.manual_seed(0)   ### for randomly selecting negative samples instead of SHEM


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import glob, os
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import pandas as pd

import utils.exp_utils as utils
# from evaluator import Evaluator
# from predictor import Predictor

from PYTORCH_dataloader import *
from skimage.morphology import ball, disk, dilation
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import pandas as pd
from skimage import measure

from sklearn.metrics import jaccard_score
from skimage import measure

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  # new thing? what do? must be True

""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""  Network Begins: """


s_path = '/media/user/8TB_HDD/Training_blocks/(2) Check_lightsheet_NO_transforms_4deep_LR_same/'


overlap_percent = 0.5
input_size = 128
depth = 16
num_truth_class = 2

""" TO LOAD OLD CHECKPOINT """
# Read in file names
onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
onlyfiles_check.sort(key = natsort_key1)

""" Find last checkpoint """       
last_file = onlyfiles_check[-1]
split = last_file.split('check_')[-1]
num_check = split.split('.')
checkpoint = num_check[0]
checkpoint = 'check_' + checkpoint
num_check = int(num_check[0])

check = torch.load(s_path + checkpoint, map_location=device)

tracker = check['tracker']
valid_idx = tracker.idx_valid

unet = check['model_type']
unet.load_state_dict(check['model_state_dict'])
unet.eval()
#unet.training # check if mode set correctly
unet.to(device)

print('parameters:', sum(param.numel() for param in unet.parameters()))
input_path = '/media/user/8TB_HDD/Training_blocks/Training_blocks_blocks_128_16_UNet/'
mean_arr = np.load(input_path + 'normalize/mean_VERIFIED.npy')
std_arr = np.load(input_path + 'normalize/std_VERIFIED.npy')       


input_path = '/media/user/8TB_HDD/Training_blocks/Training_blocks_blocks_128_16_UNet/'

""" Load filenames from tiff """
images = glob.glob(os.path.join(input_path,'*_INPUT.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
examples = [dict(input=i,truth=i.replace('_INPUT.tif','_TRUTH.tif')) for i in images]
counter = list(range(len(examples)))
val_set = Dataset_tiffs(tracker.idx_valid, examples, tracker.mean_arr, tracker.std_arr,
                                  sp_weight_bool=tracker.sp_weight_bool, transforms = 0)

val_generator = data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1,
                pin_memory=False, drop_last = True)

# def plot_max(im, ax=0, plot=1):
#      max_im = np.amax(im, axis=ax)
#      if plot:
#          plt.figure(); plt.imshow(max_im)
     
#      return max_im



#%% Import MaskRCNN
""" expand coords into a neighborhood """
def expand_coord_to_neighborhood(coords, lower, upper):
    neighborhood_be = []
    for idx in coords:
        for x in range(-lower, upper):
            for y in range(-lower, upper):
                for z in range(-lower, upper):
                    new_idx = [idx[0] + x, idx[1] + y, idx[2] + z]
                    neighborhood_be.append(new_idx)    
                    
    return neighborhood_be

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
        



""" Remove overlap between bounding boxes """
### since 3d polygons are complex, we will do 2d slice-by-slice to cut up the boxes
def split_boxes_by_Voronoi3D(box_coords, vol_shape):
    debug = 0
    
    box_ids = []
    bbox = []
    for box_id, box3d in enumerate(box_coords):
       
       #for depth in range(box3d[4], box3d[5] + 1):
          bbox.append(box3d)   ### only get x1, y1, x2, y2
       #    box_depth.append(depth)
          box_ids.append(box_id + 1)  ### cant start from zero!!!
           

    y1 = box_coords[:,0]
    x1 = box_coords[:,1]
    y2 = box_coords[:,2]
    x2 = box_coords[:,3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    #if dim == 3:
    z1 = box_coords[:, 4]
    z2 = box_coords[:, 5]
    areas *= (z2 - z1 + 1)

    
   

    centroids = [np.round((box_coords[:, 5] - box_coords[:, 4])/2 + box_coords[:, 4]), np.round((box_coords[:, 2] - box_coords[:, 0])/2 + box_coords[:, 0]), np.round((box_coords[:, 3] - box_coords[:, 1])/2 + box_coords[:, 1])]
    centroids = np.transpose(centroids)
    centroids = np.asarray(centroids, dtype=int)
    
    df = pd.DataFrame(data={'ids': box_ids, 'bbox':bbox, 'bbox_coords':[ [] for _ in range(len(box_ids)) ]})
    

    ### removes infinite lines from voronoi if you add the corners of the image!
    hack = np.asarray([[-100,-100,-100],
                        [-100, vol_shape[1]*10, -100],
                        [-100, -100, vol_shape[2]*10],
                        [-100, vol_shape[1]*10, vol_shape[2]*10],
                        
                        [vol_shape[0]*10, -100, -100],
                        [vol_shape[0]*10, vol_shape[1]*10, -100],
                        [vol_shape[0]*10, -100, vol_shape[2]*10],
                        [vol_shape[0]*10, vol_shape[1]*10, vol_shape[2]*10]
                        ])
                        
  
    centroids_vor = np.concatenate((centroids, hack))
    
    
    no_overlap = np.zeros(vol_shape)
    tmp_boxes = np.zeros(vol_shape)
    list_coords = []
    for b_id, box in enumerate(box_coords):
        
        a,b,c = np.meshgrid(np.arange(box[4], box[5]), np.arange(box[0], box[2]), np.arange(box[1], box[3]))
        
        
        ### TIGER UPDATE - shrink bounding boxes because they are expanded!!!
        # z_r = np.arange(box[4] + 1, box[5] - 1)
        # if len(z_r) == 0: z_r = box[4] + 1
        
        # x_r = np.arange(box[0] + 1, box[2] - 1)
        # if len(x_r) == 0: x_r = box[0] + 1        
        
        # y_r = np.arange(box[1] + 1, box[3] - 1)
        # if len(y_r) == 0: y_r = box[3] + 1        
        
        # a,b,c = np.meshgrid(z_r, x_r, y_r)
        
        
        coords = np.vstack([a.ravel(), b.ravel(), c.ravel()]).T  # unravel and transpose
                       
        list_coords.append(coords)
        
        
        no_overlap[coords[:, 0], coords[:, 1], coords[:, 2]] = b_id + 1  ### can't start from zero!!!
        tmp_boxes[coords[:, 0], coords[:, 1], coords[:, 2]]  = tmp_boxes[coords[:, 0], coords[:, 1], coords[:, 2]]  + 1
        
 

    intersect_ids = []
    for b_id, coords in enumerate(list_coords):
           val = np.max(tmp_boxes[coords[:, 0], coords[:, 1], coords[:, 2]])
           if val > 1:
               intersect_ids.append(b_id)
           # else:

           #     df.at[b_id, 'bbox_coords'] = coords  ### Unnecessary, because ALL boxes are added at the end with NO exceptions
                  
    
    
    from scipy.spatial import cKDTree
    voronoi_kdtree = cKDTree(centroids)  ### only split centroids of cells with overlap
    
    split_im = np.zeros(vol_shape)
    for b_id in intersect_ids:  
        
        coords = list_coords[b_id]
        test_point_dist, test_point_regions = voronoi_kdtree.query(coords, k=1)
        
        split_im[coords[:, 0], coords[:, 1], coords[:, 2]] = test_point_regions + 1  ### can't start from zero!!!
        
        #plot_max(split_im); print(b_id)
        
        #zzz
        
        #for idp, p in enumerate(coords):
        #    split_im[p[0], p[1], p[2]] = test_point_regions[idp]

    #plot_max(split_im)
    
    
    ### Now set the overlap regions to be of value in split_im
    overlap_assigned = np.copy(tmp_boxes)                           ### start with the overlap array
    overlap_assigned[tmp_boxes <= 1] = 0                            ### remove everything that is NOT overlap
    overlap_assigned[tmp_boxes > 1] = split_im[tmp_boxes > 1]       ### set all overlap regions to have value from split_im array

    overlap_assigned[tmp_boxes <= 1] = no_overlap[tmp_boxes <= 1]   ### Now add in all the rest of the boxes!!! INCLUDING PROPER NUMBER INDEXING
    
    overlap_assigned = np.asarray(overlap_assigned, dtype=int)
    
    if debug:
        plot_max(no_overlap)
        plot_max(overlap_assigned)
        
        plt.figure(); plt.imshow(no_overlap[20])
        plt.figure(); plt.imshow(overlap_assigned[20])
    
    
    cc = measure.regionprops(overlap_assigned, intensity_image=overlap_assigned)
    for b_id, cell in enumerate(cc):
        coords = cell['coords']
        
        box_id = cell['max_intensity'] - 1  ### Convert from array value back to index of array which starts from zero!!!
        
        df.at[b_id, 'bbox_coords'] = coords

    return df
  


def plot_max(im):
     ma = np.amax(im, axis=0)
     plt.figure(); plt.imshow(ma)

     

zzz
# from functional.plot_functions_CLEANED import *


# if __name__ == '__main__':
stime = time.time()

parser = argparse.ArgumentParser()

### FOR OLIGO TRAINING    
parser.add_argument('--dataset_name', type=str, default='OL_data',
                    help="path to the dataset-specific code in source_dir/datasets")



# self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024/97) new_FOV_data_det_thresh_09_check_728'

# self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM/98) new_FOV_data_det_thresh_09_NORM_check_900',

# self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM/100) nuclei_settings_Resnet50/'


# self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM/101) nuclei_settings_batch_norm/'


# self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM/102) nuclei_settings_CLEANED_DATA_AUGMENTATION/'


# self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA/104) nuclei_settings_NORM_noAug_NEW_DATA/'

# self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA/105) nuclei_settings_NOREPLACE_noAug_epochstep/'


# self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED/108) same as 104 cleaned data/'

# exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED/109) same as 104 but CLEANED OVERLAP to 04/'

# exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED/110) same as 109 but with data aug/'

exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED/111) same as 110 but with min_conf_01/'


# exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED/112) same as 111 but with NO data aug - short run/'


# exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED/113) same as 111 but NO data aug - added mask_shape_56x56x20/'
            
            
# exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED_DILATED/114) same as 111 but dilated and DATA AUG/'
            

parser.add_argument('--exp_dir', type=str, default=exp_dir,
                     help='path to experiment dir. will be created if non existent.')




# parser.add_argument('-m', '--mode', type=str,  default='train_test', help='one out of: create_exp, analysis, train, train_test, or test')
parser.add_argument('-m', '--mode', type=str,  default='test', help='one out of: create_exp, analysis, train, train_test, or test')

parser.add_argument('-f', '--folds', nargs='+', type=int, default=None, help='None runs over all folds in CV. otherwise specify list of folds.')
parser.add_argument('--server_env', default=False, action='store_true', help='change IO settings to deploy models on a cluster.')
parser.add_argument('--data_dest', type=str, default=None, help="path to final data folder if different from config")
parser.add_argument('--use_stored_settings', default=False, action='store_true',
                    help='load configs from existing exp_dir instead of source dir. always done for testing, '
                         'but can be set to true to do the same for training. useful in job scheduler environment, '
                         'where source code might change before the job actually runs.')
parser.add_argument('--resume', action="store_true", default=False,
                    help='if given, resume from checkpoint(s) of the specified folds.')
parser.add_argument('-d', '--dev', default=False, action='store_true', help="development mode: shorten everything")

args = parser.parse_args()
args.dataset_name = os.path.join("datasets", args.dataset_name) if not "datasets" in args.dataset_name else args.dataset_name
folds = args.folds
resume = None if args.resume in ['None', 'none'] else args.resume


if args.mode == 'test':
 
    cf = utils.prep_exp(args.dataset_name, args.exp_dir, args.server_env, use_stored_settings=True, is_training=False)
    if args.data_dest is not None:
        cf.data_dest = args.data_dest
    logger = utils.get_logger(cf.exp_dir, cf.server_env, cf.sysmetrics_interval)
    ### The line below prevents matplotlib plotting
    # data_loader = utils.import_module('data_loader', os.path.join(args.dataset_name, 'data_loader.py'))
    model = utils.import_module('model', cf.model_path)
    logger.info("loaded model from {}".format(cf.model_path))
    
    fold_dirs = sorted([os.path.join(cf.exp_dir, f) for f in os.listdir(cf.exp_dir) if
                 os.path.isdir(os.path.join(cf.exp_dir, f)) and f.startswith("fold")])
    if folds is None:
        folds = range(cf.n_cv_splits)
    if args.dev:
        folds = folds[:2]
        cf.max_test_patients, cf.test_n_epochs = 2, 2
    else:
        torch.backends.cudnn.benchmark = cf.cuda_benchmark
        
    # for fold in folds:
    cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(folds[0]));
    cf.fold = 0           
    logger.set_logfile(fold=cf.fold)   
    
    # -------------- inits and settings -----------------
    net = model.net(cf, logger).cuda()
    if cf.optimizer == "ADAMW":
        optimizer = torch.optim.AdamW(utils.parse_params_for_optim(net, weight_decay=cf.weight_decay,
                                                                   exclude_from_wd=cf.exclude_from_wd,
                                                                   ), 
                                      lr=cf.learning_rate[0])
    if cf.dynamic_lr_scheduling:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=cf.scheduling_mode, factor=cf.lr_decay_factor,
                                                               patience=cf.scheduling_patience)
    model_selector = utils.ModelSelector(cf, logger)
    
    
    
    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*180_cur_params.pth'))
    # onlyfiles_check.sort(key = natsort_key1)
    
    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*_best_params.pth'))
    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*800_cur_params.pth'))
    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*5300_cur_params.pth'))
    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*20_cur_params.pth'))
    
    
    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*200_cur_params.pth'))
    
    
    model_selector = utils.ModelSelector(cf, logger)

    starting_epoch = 1

    last_check = 1
    
    state_only = 0  ### for if you use "best params"
    if state_only:
        weight_path = onlyfiles_check[-1]   ### ONLY SOME CHECKPOINTS WORK FOR SOME REASON???


    onlyfiles_check.sort(key = natsort_key1)
    
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
    
            
            if state_only:
                ### also cast to device 0 --- otherwise done in exp_utils.py (with load_checkpoint() call)
                checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda(0))
                
                net.load_state_dict(checkpoint)
            
            
            optimizer = torch.optim.AdamW(utils.parse_params_for_optim(net, weight_decay=cf.weight_decay,
                                                                   exclude_from_wd=cf.exclude_from_wd,
                                                                   ), 
                                      lr=cf.learning_rate[0])        
            
            
            if not state_only:
                checkpoint_path = onlyfiles_check[-1]
            
            
                starting_epoch, net, optimizer, model_selector = \
                    utils.load_checkpoint(checkpoint_path, net, optimizer, model_selector)
                
            
            net.eval()
            net = net.cuda(device)
    
    
    
    
    
    
    all_jacc = []
    
    ### Have to change location of data due to different computer
    cf.pp_rootdir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED/OL_data/'
    
    cf.data_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED/OL_data/Tiger/train/'
            
    # batch_gen = data_loader.get_train_generators(cf, logger)
    
    ### Load for normalization
    mean_data = np.load(cf.exp_dir + "/mean_2024_MaskRCNN.npy")    
    std_data = np.load(cf.exp_dir + "/std_2024_MaskRCNN.npy")
    
  
    # checkpoint_path = onlyfiles_check[0]
    
    # checkpoint_path = os.path.join(cf.fold_dir, "last_state.pth")
    # starting_epoch, net, optimizer, model_selector = \
    #     utils.load_checkpoint(checkpoint_path, net, optimizer, model_selector)
    # logger.info('resumed from checkpoint {} to epoch {}'.format(checkpoint_path, starting_epoch))
       
    val_num = 0
    # epoch = 1
    all_jacc_check = []
    
    all_dict = []
    with torch.no_grad():
            val_results_list = []
    
            # for i in range(batch_gen['n_val']):
                                   
            #     print(i)
            #     logger.time("val_batch")
            #     batch_maskrcnn = next(batch_gen[cf.val_mode])
            #     # if cf.val_mode == 'val_patient':
            #     #     results_dict = val_predictor.predict_patient(batch)
            #     # elif cf.val_mode == 'val_sampling':
            #     #     results_dict = net.train_forward(batch, is_validation=True)
            #     zzz
             #%% UNet 
            unet.eval()
            
            # zzz
            for batch_x_val, batch_y_val, spatial_weight, input_name in val_generator:
                
                
                ### Save results for later
                
                """ Transfer to GPU to normalize ect... """
                inputs_val, labels_val = transfer_to_GPU(batch_x_val, batch_y_val, device, mean_arr, std_arr)
     
                # forward pass to check validation
                output_val = unet(inputs_val)
                
                """ Convert back to cpu """                                      
                output_tile = output_val.cpu().data.numpy()            
                output_tile = np.moveaxis(output_tile, 1, -1)
                seg_train = np.argmax(output_tile[0], axis=-1)  
                
                seg_unet = np.expand_dims(np.expand_dims(seg_train, 0), 2)
                
                input_im = batch_x_val.cpu().data.numpy()
                input_im = np.moveaxis(np.expand_dims(input_im, axis=0), 1, 2)
                truth_im = batch_y_val.cpu().data.numpy()
                truth_im = np.moveaxis(np.expand_dims(truth_im, axis=0), 1, 2)
                truth_im = np.asarray(truth_im, dtype=np.uint32)
                truth_labs = np.copy(truth_im)
                truth_im[truth_im > 0] = 1
                
                
                ### WATERSHED

                              
                # Now we want to separate the two objects in image
                # Generate the markers as local maxima of the distance to the background
                
                # if val_num == 8:
                #     zzz
                
                for_water = np.copy(seg_unet[0, :, 0, ...])
                distance = ndi.distance_transform_edt(for_water)
                coords = peak_local_max(distance, min_distance=3, 
                                        footprint=np.ones((3, 3, 3)), labels=for_water)
                mask = np.zeros(distance.shape, dtype=bool)
                mask[tuple(coords.T)] = True
                markers, _ = ndi.label(mask)
                labels = watershed(-distance, markers, mask=for_water)
                                    
                labels_unet = np.expand_dims(np.expand_dims(labels, 0), 2)        
                                    
                
                
                  
                #%% Arrange data for MaskRCNN
                
                # zzz
                ### This also prevents plotting?
                # from inference_utils import *
                
                batch_size = 1
                patch_size = input_size
                patch_depth = depth
                batch_im = np.moveaxis(input_im, 1, -1)
                
                ### NORMALIZE
                batch_im = (batch_im - mean_data)/std_data
                
                batch = {'data':batch_im, 'seg': np.zeros([batch_size, 1, patch_size, patch_size, patch_depth]), 
                          'class_targets': np.asarray([]), 'bb_targets': np.asarray([]), 
                          'roi_masks': np.zeros([batch_size, 1, 1, patch_size, patch_size, patch_depth]),
                          'patient_bb_target': np.asarray([]), 'original_img_shape': batch_im.shape,
                          'patient_class_targets': np.asarray([]), 'pid': ['0']}
                
                
                output = net.test_forward(batch, main_brain=True) #seg preds are only seg_logits! need to take argmax.
                            

                                      
                results_dict = net.get_results(img_shape=output[0], detections=output[1], 
                                               detection_masks=output[2], return_masks=output[3]) 
                im_shape = output[0]
                
                
                if 'seg_preds' in results_dict.keys():
                      results_dict['seg_preds'] = np.argmax(results_dict['seg_preds'], axis=1)[:,np.newaxis]
                    # results_dict['colored_boxes'] = np.expand_dims(results_dict['colored_boxes'][:, 1, :, :, :], axis=0)
                     
                ### Add to segmentation
                seg_maskrcnn = np.moveaxis(results_dict['seg_preds'], -1, 1) 
                                
                
                                
                if val_num == 8:
                    zzz
                
 
                ### Add box patch factor
                new_box_list = []
                tmp_check = np.zeros(im_shape[2:])
                
                thresh = 0.9
                for bid, box in enumerate(results_dict['boxes'][0]):
                      """ 
                      
                          Some bounding boxes have no associated segmentations!!! Skip these

                      """
                      if len(box['mask_coords']) == 0:
                          continue
                                                         
                      ### Only add if above threshold
                      # print(box['box_score'] )
                      if box['box_score'] > thresh:
                          new_box_list.append(box)     
                        
                      
                results_dict['boxes'] = [new_box_list]
                
                if len(results_dict['boxes'][0]) > 0:
                    print('Splitting boxes with KNN')
                    box_vert = []
                    for box in results_dict['boxes'] [0]:
                        box_vert.append(box['box_coords'])

                    box_vert = np.asarray(np.round(np.vstack(box_vert)), dtype=int)
                    
                    
                    seg_overall = seg_maskrcnn[0, :, 0, ...]
                    
                    
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

                    new_labels[seg_overall == 0] = 0   ### This is way simpler and faster than old method of looping through each detection
                    #plot_max(new_labels)
                    
                    
                    print('Step 1: Cleaning up spurious assignments')         
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

                        bw_lab = measure.label(tmp, connectivity=1)
                        
                        
                        if np.max(bw_lab) > 1:
                            check_cc = measure.regionprops(bw_lab)

                            #print(str(id_o))
                            all_lens = []
                            all_coords = []
                            for check in check_cc:
                                all_lens.append(len(check['coords']))
                                all_coords.append(check['coords'])
                            
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
  
                    ### Expand each to_assign to become a neighborhood!  ### OR JUST DILATE THE WHOLE IMAGE?
                    clean_labels = new_labels
                    clean_labels[to_assign > 0] = 0

                    clean_labels = expand_add_stragglers(to_assign, clean_labels)

                    
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

                    #small = np.asarray(small, np.int32)
                    clean_labels[small > 0] = 0   ### must remember to mask out all the small areas otherwise will get reassociated back with the small area!
                    clean_labels = expand_add_stragglers(small, clean_labels)

                    ### Add back in all the small objects that were NOT near enough to touch anything else
                    small[clean_labels > 0] = 0
                    clean_labels[small > 0] = small[small > 0]


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
                                
                    
                    
                    
                    #%% WITHOUT the moving FOV analysis a lot of detections that are under bounding box thresh are tossed
                    ### so let's try to add those segmentations back in
                    
                    
                    bw_coloc = np.copy(clean_labels)
                    bw_coloc[bw_coloc > 0] = 1
                    bw_coloc = bw_coloc + seg_maskrcnn[0, :, 0, ...]
                    
                    labs_coloc = measure.label(bw_coloc > 0)
                    cc_coloc = measure.regionprops(labs_coloc, intensity_image=bw_coloc)
                    
                    obj_val = np.max(clean_labels)
                    for obj in cc_coloc:
                        
                        max_val = obj['max_intensity']
                        if max_val > 1:  ### continue if not stand-alone object
                            # print('skip')
                            continue
                        else:
                            coords = obj['coords']
                            clean_labels[coords[:, 0], coords[:, 1], coords[:, 2]] = obj_val
                            obj_val += 1
                            
                    
                    leftovers = np.copy(seg_maskrcnn[0, :, 0, ...])
                    leftovers[clean_labels > 0] = 0
                    left = measure.label(leftovers)
                    left_cc = measure.regionprops(left)
                    
                    for obj in left_cc:
                        
                        coords = obj['coords']
                        if len(coords) > 50:
                            clean_labels[coords[:, 0], coords[:, 1], coords[:, 2]] = obj_val
                            obj_val += 1                            
                    
                    
                    
                    
                    labels_maskrcnn = np.asarray(clean_labels, np.int32)
                    
                    
                                        
                    ### Add dilation by ball
                    """ dilates image by a spherical ball of size radius """
                    def dilate_by_ball_to_grayscale(input_im, radius):
                          ball_obj = ball(radius=radius)
                          input_im = dilation(input_im, footprint=ball_obj)  
                          #input_im[input_im > 0] = 1
                          return input_im
                
                    """ dilates image by a spherical ball of size radius """
                    def dilate_by_disk_to_binary(input_im, radius):
                          ball_obj = disk(radius=radius)
                          for s_id in range(len(input_im)):
                              input_im[s_id] = dilation(input_im[s_id], footprint=ball_obj)  
                          return input_im

                
                    # labels_maskrcnn = dilate_by_ball_to_grayscale(labels_maskrcnn, radius=1)
                    labels_maskrcnn = dilate_by_disk_to_binary(labels_maskrcnn, radius=1)
                    
                    
                    
                    
                    labels_maskrcnn = np.expand_dims(np.expand_dims(labels_maskrcnn, 0), 2)
                    
                    


                else:
                    labels_maskrcnn = np.zeros(np.shape(input_im))
                    labels_maskrcnn = np.asarray(labels_maskrcnn, dtype=np.uint32)
                


            
            
                #%% Calculate our own stats
                dict_results = {'filename':input_name, 
                                'OVERALL_jaccard_unet':0, 'object_jaccard_unet':0, 'sensitivity_unet':0, 'precision_unet':0,
                                                'TP_unet':0, 'TN_unet':0, 'FP_unet':0, 'FN_unet':0,
                                'OVERALL_jaccard_maskRCNN':0, 'object_jaccard_maskRCNN':0, 'sensitivity_maskRCNN':0, 'precision_maskRCNN':0,
                                                'TP_maskRCNN':0, 'TN_maskRCNN':0, 'FP_maskRCNN':0, 'FN_maskRCNN':0}
                
                

                out_file = args.exp_dir + 'output_batch/'
                
                

                if len(np.where(truth_im > 0)[0]) > 0 or len(np.where(labels_unet > 0)[0]) > 0:
                                                
                    print(val_num)
                    def calculate_metrics(truth_labs, labels):
                        
                        truth_bw = truth_labs > 0
                        seg_bw = labels > 0

                        iou = jaccard_score(truth_bw.flatten(), seg_bw.flatten())
                        
                        all_jacc_check.append(iou)
                        
                        dict_results['jaccard_unet'] = iou

                        
                        ### Find TP, FP, FN
                        # lab_truth = measure.label(truth_bw[0, :, 0, ...])
                        cc_truth = measure.regionprops(truth_labs)
                        
                        cc_seg = measure.regionprops(labels[0, :, 0,...])
                        
                        # Loop through each prediction and find out if it is matched with truth, using iou thresh of... 0.5?
                        iou_thresh = 0.3
                        TP = FP = FN = 0
                        truth_ids_match = []
                        obj_IoU = []
                        for cell in cc_seg:
                            # check_cell = np.zeros(np.shape(labels_unet))
                            c = cell['coords']
                            
                            # check_cell[c[:, 0], c[:, 1], c[:, 2]] = 1
                            all_match = []
                            for t_id, truth_c in enumerate(cc_truth):
                                
                                tc = truth_c['coords']
                                
                                l1 = [frozenset(i) for i in c]
                                l2 = [frozenset(i) for i in tc]
                                
                                
                                TP_iou = len(set(l2)&set(l1))
                                FP_iou = len(l1) - TP_iou
                                FN_iou = len(l2) - TP_iou
                                
                                IoU = TP_iou / (TP_iou + FP_iou + FN_iou)
                                
                                if IoU > iou_thresh:
                                    all_match.append(t_id)
                                    
                                    truth_ids_match.append(t_id)
                                    obj_IoU.append(IoU)
                            ### In case there is more than 1 match
                            if len(all_match) > 1:
                                
                                print('MULTI-MATCH')
                                # zzz
                                
                            ### Else start counting
                            if len(all_match) > 0:
                                TP += 1
                            else:
                                FP += 1
                        
                        FN += len(cc_truth) - len(np.unique(truth_ids_match))
                        
                        return iou, obj_IoU, TP, FP, FN
                        
                    
                    #%% Run for UNet
                    iou, obj_IoU, TP, FP, FN = calculate_metrics(truth_labs[0, :, 0, ...], labels_unet)
                    dict_results['TP_unet'] = TP
                    dict_results['FP_unet'] = FP
                    dict_results['FN_unet'] = FN
                    dict_results['object_jaccard_unet'] = obj_IoU
                    if TP + FN > 0: dict_results['sensitivity_unet'] = TP/(TP + FN)
                    if TP + FP > 0: dict_results['precision_unet'] = TP/(TP + FP)
                    dict_results['OVERALL_jaccard_unet'] = iou
                    
                    
                    
                    #%% Run for MaskRCNN

                    iou, obj_IoU, TP, FP, FN = calculate_metrics(truth_labs[0, :, 0, ...], labels_maskrcnn)
                    dict_results['TP_maskRCNN'] = TP
                    dict_results['FP_maskRCNN'] = FP
                    dict_results['FN_maskRCNN'] = FN
                    dict_results['object_jaccard_maskRCNN'] = obj_IoU
                    if TP + FN > 0: dict_results['sensitivity_maskRCNN'] = TP/(TP + FN)
                    if TP + FP > 0: dict_results['precision_maskRCNN'] = TP/(TP + FP)
                    dict_results['OVERALL_jaccard_maskRCNN'] = iou
                    
                    
                    
                    all_dict.append(dict_results) 
    
                    #----------- Plot same image each time -------------#
                                    
                    # if val_num < 20 or val_num == 121:
                    ### plot concatenated TIFF
                    # truth_im[truth_im > 0] = 65535
                    seg_maskrcnn[seg_maskrcnn > 0] = 65535
                    # seg_unet[seg_unet > 0] = 65535
                    
                    concat  = np.concatenate((np.asarray(input_im, dtype=np.uint16), np.asarray(truth_labs, dtype=np.uint16), 
                                              np.asarray(seg_maskrcnn, dtype=np.uint16), np.asarray(labels_maskrcnn, dtype=np.uint16),
                                              np.asarray(labels_unet, dtype=np.uint16)))
            
                    concat = np.moveaxis(concat, 0, 2)       
                    concat = np.moveaxis(concat, 0, 1)                         
                        
                    tiff.imwrite(out_file + str(val_num) + '_VAL_IM_batch_' +  str(0)  + '_COMPOSITE.tif', concat,
                                  imagej=True,   metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})
            
            
                # if val_num == 8:
                #     zzz
                val_num += 1
                
                
                    
                # else:
                #     zzz
                        
                        
    
    len(all_dict)

    df_all = pd.DataFrame(all_dict)  
    
    df_maskrcnn = df_all[['sensitivity_maskRCNN', 'precision_maskRCNN','OVERALL_jaccard_maskRCNN',
                          'TP_maskRCNN', 'FN_maskRCNN', 'FP_maskRCNN', 'object_jaccard_maskRCNN']]
    df_maskrcnn['model_name'] = 'maskrcnn'
    df_maskrcnn = df_maskrcnn.rename(columns={'sensitivity_maskRCNN':'Sensitivity', 'precision_maskRCNN':'Precision','OVERALL_jaccard_maskRCNN':'Jaccard',
                          'TP_maskRCNN':'TP', 'FN_maskRCNN':'FN', 'FP_maskRCNN':'FP', 'object_jaccard_maskRCNN':'jacc_obj'})
    
    
    df_unet = df_all[['sensitivity_unet', 'precision_unet','OVERALL_jaccard_unet',
                          'TP_unet', 'FN_unet', 'FP_unet', 'object_jaccard_unet']]    

    df_unet['model_name'] = 'unet'
    df_unet = df_unet.rename(columns={'sensitivity_unet':'Sensitivity', 'precision_unet':'Precision','OVERALL_jaccard_unet':'Jaccard',
                          'TP_unet':'TP', 'FN_unet':'FN', 'FP_unet':'FP', 'object_jaccard_unet':'jacc_obj'})
    

    df_combined = pd.concat([df_maskrcnn, df_unet])
    
    df_combined.to_pickle(out_file + 'compare_pickle_output_DILATE_slice_updated_thresh03.pkl')
    
    
    
    
    
                 
    print(np.mean(df_all['sensitivity_maskRCNN'].iloc[np.where(df_all['sensitivity_maskRCNN'] > 0)[0]]))
    print(np.mean(df_all['precision_maskRCNN'].iloc[np.where(df_all['precision_maskRCNN'] > 0)[0]]))
    print(np.mean(df_all['OVERALL_jaccard_maskRCNN'].iloc[np.where(df_all['OVERALL_jaccard_maskRCNN'] > 0)[0]]))


    np.sum(df_all['TP_maskRCNN'])/(np.sum(df_all['TP_maskRCNN']) + np.sum(df_all['FN_maskRCNN']))
    np.sum(df_all['TP_maskRCNN'])/(np.sum(df_all['TP_maskRCNN']) + np.sum(df_all['FP_maskRCNN']))

    print(np.mean(df_all['sensitivity_unet'].iloc[np.where(df_all['sensitivity_unet'] > 0)[0]]))
    print(np.mean(df_all['precision_unet'].iloc[np.where(df_all['precision_unet'] > 0)[0]]))
    print(np.mean(df_all['OVERALL_jaccard_unet'].iloc[np.where(df_all['OVERALL_jaccard_unet'] > 0)[0]]))
    
    np.sum(df_all['TP_unet'])/(np.sum(df_all['TP_unet']) + np.sum(df_all['FN_unet']))
    np.sum(df_all['TP_unet'])/(np.sum(df_all['TP_unet']) + np.sum(df_all['FP_unet']))

    

                    # else:
                    #     zzz
                        
        # all_jacc.append(all_jacc_check)




