#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:54:13 2023

@author: user
"""


from functional.tree_functions import *
import os
import glob
import apoc
import z5py
import concurrent.futures
from tqdm import tqdm
import time
from predictor import apply_wbc_to_patient
from functional.matlab_crop_function import *
from scipy.stats import norm
from multiprocessing.pool import ThreadPool
from inference_analysis_OL_TIFFs_small_patch_POOL import post_process_async, expand_add_stragglers
from inference_utils import *
from skimage.measure import label, regionprops, regionprops_table
from tkinter import filedialog
import tkinter
from tifffile import *
from os.path import isfile, join
from os import listdir
from natsort import natsort_keygen, ns
import numpy as np
import pandas as pd
import torch

import utils.exp_utils as utils
# import utils.model_utils as mutils

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

natsort_key1 = natsort_keygen(
    key=lambda y: y.lower())      # natural sorting order


# import h5py


# import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def train_APOC_blood():
    im1 = imread(
        '/media/user/8TB_HDD/Autofluor_blood_APOC_seg/otx18_MoE_2yo_fused_302_green_CROP.tif')
    im2 = imread(
        '/media/user/8TB_HDD/Autofluor_blood_APOC_seg/otx18_MoE_2yo_fused_302_red_CROP.tif')
    labs = imread(
        '/media/user/8TB_HDD/Autofluor_blood_APOC_seg/otx18_MoE_2yo_fused_302_LABELS.tif')

    features = "original gaussian_blur=1 laplace_box_of_gaussian_blur=1"

    save_path = '/media/user/8TB_HDD/Autofluor_blood_APOC_seg/blood_classifier'
    pixel_classifier = apoc.PixelClassifier(opencl_filename=save_path)

    pixel_classifier.train(features=features,
                           ground_truth=labs,
                           image=[im1, im2])

    # continue training with more data!!!
    im1 = imread(
        '/media/user/8TB_HDD/Autofluor_blood_APOC_seg/otx18_MoE_2yo_fused_302_green_CROP_LOWER.tif')
    im2 = imread(
        '/media/user/8TB_HDD/Autofluor_blood_APOC_seg/otx18_MoE_2yo_fused_302_red_CROP_LOWER.tif')
    labs = imread(
        '/media/user/8TB_HDD/Autofluor_blood_APOC_seg/otx18_MoE_2yo_fused_302_CROP_LOWER_LABELS.tif')
    pixel_classifier.train(features=features,
                           ground_truth=labs,
                           image=[im1, im2],
                           continue_training=True)

    # continue training with more data!!!
    im1 = imread(
        '/media/user/8TB_HDD/Autofluor_blood_APOC_seg/otx18_MoE_2yo_fused_311_green_CROP.tif')
    im2 = imread(
        '/media/user/8TB_HDD/Autofluor_blood_APOC_seg/otx18_MoE_2yo_fused_311_red_CROP.tif')
    labs = imread(
        '/media/user/8TB_HDD/Autofluor_blood_APOC_seg/otx18_MoE_2yo_fused_311_LABELS.tif')
    pixel_classifier.train(features=features,
                           ground_truth=labs,
                           image=[im1, im2],
                           continue_training=True)

    # semantic_segmentation = pixel_classifier.predict(image=[im1, im2])

    return pixel_classifier


def APOC_by_crop(pixel_classifier, ch1_im, ch2_im, quad_depth, quad_size):
    im_size = np.shape(ch1_im)

    width = im_size[1]
    height = im_size[2]
    depth_im = im_size[0]

    seg_im = np.zeros(np.shape(ch1_im))
    seg_no_clean = np.zeros(np.shape(ch1_im))
    for z in range(0, depth_im + quad_depth, round(quad_depth)):

        if z + quad_depth > depth_im:
            print('reached end of dim')
            continue
        for x in range(0, width + quad_size, round(quad_size)):
            if x + quad_size > width:
                print('reached end of dim')
                continue
            for y in range(0, height + quad_size, round(quad_size)):
                if y + quad_size > height:
                    print('reached end of dim')
                    continue

                print([x, y, z])

                ch1_crop = ch1_im[z:z + quad_depth,
                                  x:x + quad_size, y:y + quad_size]
                ch2_crop = ch2_im[z:z + quad_depth,
                                  x:x + quad_size, y:y + quad_size]

                seg_test = pixel_classifier.predict(image=[ch1_crop, ch2_crop])

                seg_test = np.asarray(seg_test)

                seg_test[seg_test < 2] = 0
                seg_test[seg_test > 0] = 1

                # Remove small blobs
                labs = morphology.label(seg_test)
                cc = measure.regionprops(labs)
                med_size = 300
                ratio_thresh = 5  # closer to 0 is a circle

                min_size = 100

                clean_arr = np.zeros(np.shape(seg_test))
                for obj in cc:

                    vol = len(obj['coords'])

                    coords = obj['coords']

                    if vol > med_size:  # keep anything quite big
                        clean_arr[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

                    if vol > min_size:
                        major = obj['axis_major_length']
                        try:
                            minor = obj['axis_minor_length']
                        except:
                            minor = 0
                        if minor == 0:  # means super long
                            print('long object')
                        else:
                            ratio = major/minor

                        if vol > min_size and vol < med_size and ratio > ratio_thresh:
                            clean_arr[coords[:, 0],
                                      coords[:, 1], coords[:, 2]] = 1

                # clean_arr = seg_test

                # ### save
                # seg_im[z:z+quad_depth, x:x+quad_size, y:y+quad_size] = clean_arr
                seg_no_clean[z:z+quad_depth, x:x +
                             quad_size, y:y+quad_size] = clean_arr

    return seg_no_clean


# ANYA -
if __name__ == "__main__":
    class Args():
        def __init__(self):

            self.dataset_name = "datasets/OL_data"
            # self.exp_dir = '/media/user/FantomHD/Lightsheet data/Training_data_lightsheet/Training_blocks/Training_blocks_RegRCNN/94) newest_CLEANED_shrunk_det_thresh_02_min_conf_01/'
            # self.exp_dir = '/media/user/fa2f9451-069e-4e2f-a29b-3f1f8fb64947/Training_checkpoints_RegRCNN/96) new_FOV_data_det_thresh_09_check_300'
            # self.exp_dir = '/media/user/8TB_HDD/Training_checkpoints_RegRCNN/96) new_FOV_data_det_thresh_09_check_300'

            # self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024/97) new_FOV_data_det_thresh_09_check_728'

            # self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM/98) new_FOV_data_det_thresh_09_NORM_check_900',

            # self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM/100) nuclei_settings_Resnet50/'

            # self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM/101) nuclei_settings_batch_norm/'

            # self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM/102) nuclei_settings_CLEANED_DATA_AUGMENTATION/'

            # self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA/104) nuclei_settings_NORM_noAug_NEW_DATA/'

            # self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA/105) nuclei_settings_NOREPLACE_noAug_epochstep/'

            # self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED/108) same as 104 cleaned data/'

            # Was pretty good! A little under-sensitive on the bottom slices
            # self.exp_dir = '/media/user/8TB_HDD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED/110) same as 109 but with data aug/'

            # JAIME
            # Tiger --- Try this out with Jaime
            self.exp_dir = '/media/user/4TB_SSD/Training_blocks_RegRCNN_UPDATED_2024_NORM_ADDED_DATA_CLEANED/111) same as 110 but with min_conf_01/'

            # self.exp_dir = '/media/user/8TB_HDD/Training_data_96rik/7) UPDATED_training_data_cleaned'

            # ANYA -
            # self.exp_dir = '/media/user/8TB_HDD/Training_data_96rik/Training_data_RegRCNN_UPDATED_2024_NORM/8) updated_96rik_with_NORM'  ### Tiger --- Anya I commented this out
            # self.exp_dir = '/media/user/20TB_HDD_NAS_2/Training_data_96rik/Training_data_RegRCNN_UPDATED_2024_NORM/8) updated_96rik_with_NORM'
            # moved to above because of drive slot issue
            self.server_env = False

    args = Args()
    data_loader = utils.import_module(
        'dl', os.path.join(args.dataset_name, "data_loader.py"))

    config_file = utils.import_module(
        'cf', os.path.join(args.exp_dir, "configs.py"))
    cf = config_file.Configs()
    cf.exp_dir = args.exp_dir
    cf.test_dir = cf.exp_dir

    # Load for normalization
    mean_data = np.load(cf.exp_dir + "/mean_2024_MaskRCNN.npy")
    std_data = np.load(cf.exp_dir + "/std_2024_MaskRCNN.npy")

    # zzz

    cf.fold = 0
    if cf.dim == 2:
        cf.merge_2D_to_3D_preds = True
        if cf.merge_2D_to_3D_preds:
            cf.dim == 3
    else:
        cf.merge_2D_to_3D_preds = False

    cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(cf.fold))
    anal_dir = os.path.join(cf.exp_dir, "inference_analysis")

    logger = utils.get_logger(cf.exp_dir)

    # ^^^this causing a lot of output statements later

    model = utils.import_module('model', os.path.join(cf.exp_dir, "model.py"))

    # zzz
    torch.backends.cudnn.benchmark = cf.dim == 3

    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    # JAIME
    # USE THIS FOR 111) --- Oligo analysis
    onlyfiles_check = glob.glob(os.path.join(
        cf.fold_dir + '/', '*_best_params.pth'))

    # ANYA -
    # onlyfiles_check = glob.glob(os.path.join(
    # cf.fold_dir + '/', '*700_cur_params.pth'))  # USE THIS FOR 96rik

    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*5300_cur_params.pth'))
    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*2180_cur_params.pth'))

    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*160_cur_params.pth'))
    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*238_best_params.pth'))
    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*2200_cur_params.pth'))

    # use 200 for 110)
    # onlyfiles_check = glob.glob(os.path.join(cf.fold_dir + '/','*200_cur_params.pth'))

    model_selector = utils.ModelSelector(cf, logger)

    starting_epoch = 1

    last_check = 0

    # ANYA - change to 0
    # JAIME - change to 1
    state_only = 1  # for if you use "best params" should be val 1 if best params
    if state_only:
        # ONLY SOME CHECKPOINTS WORK FOR SOME REASON???
        weight_path = onlyfiles_check[-1]

    onlyfiles_check.sort(key=natsort_key1)

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
                utils.load_checkpoint(
                    checkpoint_path, net, optimizer, model_selector)

            net.eval()
            net.cuda(device)

        else:

            if state_only:
                # also cast to device 0 --- otherwise done in exp_utils.py (with load_checkpoint() call)
                checkpoint = torch.load(
                    weight_path, map_location=lambda storage, loc: storage.cuda(0))

                net.load_state_dict(checkpoint)

            optimizer = torch.optim.AdamW(utils.parse_params_for_optim(net, weight_decay=cf.weight_decay,
                                                                       exclude_from_wd=cf.exclude_from_wd,
                                                                       ),
                                          lr=cf.learning_rate[0])

            if not state_only:
                checkpoint_path = onlyfiles_check[-1]

                starting_epoch, net, optimizer, model_selector = \
                    utils.load_checkpoint(
                        checkpoint_path, net, optimizer, model_selector)

            net.eval()
            net = net.cuda(device)

    """ Select multiple folders for analysis AND creates new subfolder for results output """
    root = tkinter.Tk()
    # get input folders
    another_folder = 'y'
    list_folder = []
    input_path = "./"

    # initial_dir = '/media/user/storage/Data/'
    # while(another_folder == 'y'):
    #     input_path = filedialog.askdirectory(parent=root, initialdir= initial_dir,
    #                                         title='Please select input directory')
    #     input_path = input_path + '/'

    #     print('Do you want to select another folder? (y/n)')
    #     another_folder = input();   # currently hangs forever
    #     #another_folder = 'y';

    #     list_folder.append(input_path)
    #     initial_dir = input_path

    # M125 - is dimmer
    # M120 - is super bright somehow
    # M123 - also dimmer

    list_folder = [


        # 4mos - tiny bit oily in minor areas --- was redone later again as well

        # DECENT AMOUNT OF TISSUE TEARING AS WELL ### also a little bloody
        # '/media/user/20TB_HDD_NAS_2/20240416_M326_REDO_MoE_4mos_SHIELD_CUBIC_7d_RIMS_RI_1496_3d_sunflow_REDO_100perc_488laser_60perc_638laser_80msec/',

        # M138 - good to redo, delip, was high first time
        # '/media/user/8TB_HDD/20231116_M138_MoE_CasprtdT_Cup_CONTROL_6wk__SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER/',

        # M325 - seems okay, left hemisphere a bit smaller than right
        # '/media/user/20TB_HDD_NAS_2/20240423_M325_REDO_4mos_MoE_little_smaller_SHIELD_CUBIC_RIMS_RI_1496_4d_after_PB_wash_100perc_488_50perc_638_100msec_2/',





        # SWAPPED TO 111) check 246 after this point to be safe... should be no big diff...



        # 2mos
        # RE-RUN with more sensitive models??? (compared 110 and 111 and no real difference)
        # '/media/user/20TB_HDD_NAS_2/20240425_M260_MoE_P56_SHIELD_CUBIC_RIMS_RI_1493_fresh_100perc_488_30perc_638_80msec/',


        # --- need to run (DONE)
        # '/media/user/8TB_HDD/20231012_M223_MoE_Ai9_SHIELD_CUBIC_RIMS_RI_1500_3days_5x/',

        # --- need to run (DONE)
        # '/media/user/20TB_HDD_NAS/20240124_M299_MoE_P60_SHIELD_CUBIC7d_RIMS_RI_1489_sunflow_laser_80perc/',


        # POSTPROCESS M223 along with all of these^^^ (DONE) --- did it

        # P60 with HIGH AAVs - 3 channels!!!  (RE-RUNNING NOW WITH CHECK 111) DONE
        # '/media/user/20TB_HDD_NAS/20240210_M256_MoE_P60_high_AAVs_SHIELD_CUBIC_7d_RIMS_2d_RI_1493_sunflow_60perc/',



        # '/media/user/20TB_HDD_NAS/20240103_M126_MoE_Reuse_delip_RIMS_RI_14926_sunflow_80perc/',

        # '/media/user/8TB_HDD/20231216_M127_MoE_P56_delip_RIMS_RI_1489_sunflower_80perc/',


        # '/media/user/20TB_HDD_NAS/20240210_M254_MoE_P60_low_AAVs_SHIELD_CUBIC_7d_RIMS_2d_RI_1493_sunflower/',



        # 4mos
        # maybe re-run M138? Was defs missing some cells at midline

        # Too YOUNG P100   --- done
        # M226 - okay, delip, some tearing in cortex (didnt check for bleed through)
        # '/media/user/20TB_HDD_NAS/20240102_M226_MoE_Tie2Cre_Ai9_3mos_delip_RIMS_RI_14926_sunflow_80perc/',


        # TOO YOUNG P100 --- (done)
        # '/media/user/20TB_HDD_NAS/20240215_M264_Cup_CONTROL_3mos_SHIELD_DELIP_7d_RIMS_RI_1493_sunflow/',


        # 8 months

        # Looks clean --- (done)
        # '/media/user/20TB_HDD_NAS/20240228_M281_MoE_8mos_SHIELD_CUBIC_6d_RIMS_1d_RI_1493_reuse_sunflow_561_laser_AUTOFLUOR/',

        # Small tear near front, otherwise good (done)
        # '/media/user/20TB_HDD_NAS/20240125_M279_MoE_8mos_SHIELD_delip8d_RIMS_RI_1492_sunflow_laser_80perc/',

        # 8 months --- need to do --- quite bloody, sub vasculature --- tear in right hemisphere frontal/orbital cortex (current)
        # '/media/user/20TB_HDD_NAS/20240227_M285_MoE_8months_SHIELD_CUBIC_7d_RIMS_RI_1493_sunflow/',

        # '/media/user/20TB_HDD_NAS/20240210_M286_MoE_8mos_SHIELD_CUBIC_7d_RIMS_2d_RI_1493_sunflow_80perc/',

        # 22 mos
        # !!!
        # done --- should redo after with vessel subtraction to see difference! (DONE 1 time now - with NO vessel subtraction)
        # currently --- RE-doing WITH vessel subtraction
        # '/media/user/8TB_HDD/20240101_M91_OLD_MoE_delip_13d_CUBIC_7d_RIMS_RI_1489_sunflow_80perc_NO_FUSION/',


        # M97 - 22mos --- (DONE) - NOT bloody (no vessel sub) tear in right hemisphere somato-motor regions
        # '/media/user/20TB_HDD_NAS_2/20240308_M97_MoE_SHIELD_CUBIC_RIMS_RI_14925_14d_sunflow/',

        #  22mos - no vessel sub (RE-RUNNING NOW WITH CHECK 111)
        # '/media/user/20TB_HDD_NAS/20240127_M271_MoE_OLD_22mos_SHIELD_CUBIC10d_RIMS_RI_1493_sunflow_laser_80perc/',

        # Excellent, new brain, no blood (RE-RUNNING NOW WITH CHECK 111) - no vessel sub
        # '/media/user/4TB_SSD/20240501_M334_MoE_22mos_SHIELD_CUBIC_14d_RIMS_RI_1493_100p488_20p638_100msec/',




        # 30mos - 1Otx7 - RUN WITH NEW TRAINED MASKRCNN  *** AND with blood vessel subtraction
        # SHOULD redo with 111) check 267 instead of 110) because currently lower on cell counts

        # RE-RUNNING with 111) now (current)
        # '/media/user/20TB_HDD_NAS_2/20240214_1Otx7_Jaime-Tx_2yo_MoE_OLD_SHIELD_CUBIC_12d_RIMS_RI_1493_sunflow/',


        # ### OLD - needs vessel subtraction!!! - NEED TO DO! (DONE)
        # '/media/user/20TB_HDD_NAS_2/20240308_otx18_MoE_2yo_SHIELD_CUBIC_RIMS_RI_14925_14d_sunflow/',



        # OLD --- not terrible vessels bleed through... likely still best to do vessel dampening
        # (current)
        # '/media/user/20TB_HDD_NAS_2/20240307_5otx5_MoE_2yo_SHIELD_CUBIC_RIMS_RI_14925_14d_sunflow/',




        # 34 mos --- this is the oldest brain actually!!! --- can pool with 30mos I think...

        # new brain 1Otx6 - 30mos --- looks good, should do vessel subtraction but nowhere near as bad as 1Otx7 (current)
        # '/media/user/20TB_HDD_NAS_2/20240419_Otx6_MoE_34mos_SHIELD_CUBIC_14d_RIMS_RI_14968_100perc_488_30perc_638_100msec_6x8_tiles/'



        # CUPRIZONE + RECOVERY

        # DONE - good (DONE)
        # '/media/user/20TB_HDD_NAS_2/20240419_M312_REDO_REDO_PB_washed_GOOD_SHIELD_CUBIC_7d_RI_RIMS_14968_100perc_488_50perc_638_100msec/',

        # # ### 6wks cup + 3wks recovery (DONE)
        # '/media/user/20TB_HDD_NAS_2/20240420_M310_REDO_REDO_PB_washed_MoE_CUP_6wks_RECOV_3wks_RI_RIMS_1496_2d_after_wash_100perc_488_60perc_638_100msec/',

        # 6wks cup + 3wks recovery (DONE) - no blood vessel sub
        # '/media/user/20TB_HDD_NAS_2/20240426_M313_REDO_REDO_MoE_CUP_6wks_RECOV_3wks_SHIELD_CUBIC_RIMS_RI_1493_after_PBwash_100p488_60p638_100msec/',


        # for CUPRIZONE (DONE) - pretty good, little bit of blood vessels in frontal cortex but fine mostly --- NO vessel sub
        # '/media/user/20TB_HDD_NAS/20240227_M265_Cuprizone_6wks_SHIELD_CUBIC_7d_RIMS_RI_1493_reuse_sunflow/'

        # CUPRIZONE --- next to analyze - no vessel sub   (DONE)
        # '/media/user/20TB_HDD_NAS/20240215_M266_MoE_CUPRIZONE_6wks_SHIELD_CUBIC_7d_RIMS_RI_1493_sunflow/',




        # 6wks cuprizone --- is bloody?... should do vessel sub??? but be careful... dim vessels???
        # YES - doing vessel sub
        # '/media/user/20TB_HDD_NAS_2/20240422_M267_REDO_REDO_MoE_CUP_6wks_SHIELD_CUBIC_RIMS_RI_1496_2d_after_PB_wash_100perc_488_60perc_638_100msec/',


        # FVB --- ripped right RSP, dim autofluor, a little bloody, but should be okay because dim (NO vessel sub)
        # (DONE)
        # '/media/user/8TB_HDD/20231114_M147_MoE_FVB_7days_delip_RIMS_RI_1487_5days_5x_60perc_laser_SUNFLOWER/',


        # FVB --- good, ripped right RSP, dim autofluor (NO vessel sub) (DONE) -- need post-process
        # '/media/user/20TB_HDD_NAS/20240124_M155_FVB_MoE_SHIELD_delip7d_RIMS_PB_wash_CUBIC_2d_RIMS_RI_1489_sunflow_laser_60-60perc/',


        # FVB --- good, ripped right RSP, dim autofluor (NO vessel sub) (DONE) -- need post-process
        # '/media/user/20TB_HDD_NAS/20240125_M152_FVB_MoE_SHIELD_delip_RIMS_RI_1492_sunflower_80perc/',

        # CD1 --- good (NO vessel sub) (DONE)
        # '/media/user/20TB_HDD_NAS/20240116_M172_CD1_MoE_SHIELD_delip_RIMS_RI_1489_sunflow_laser_70-60perc/',


        # CD1 --- good (NO vessel sub) (DONE)
        # '/media/user/8TB_HDD/20231216_M170_MoE_CD1_delip_RIMS_RI_1489_sunflower_80perc/',

        # Do bent CD1??? --- DO SOON!!! - does this need autofluor sub?
        # '/media/user/20TB_HDD_NAS_2/20231116_M169_MoE_CD1_SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER_WARPED/'



        # FVB --- (current), YES vessel sub
        # --- ripped right RSP, dim autofluor, fairly bloody but dim blood (YES vessel sub --- be careful, not perfectly aligned)
        # '/media/user/20TB_HDD_NAS/20240125_M146_FVB_MoE_SHIELD_delip8d_RIMS_RI_1492_sunflow_laser_80perc/',




        # 6wks cup + 3wks recovery (DONE)
        # A little bit wonky? Lower numbers... was also the first to be redone with PB wash... and probably not enough time in RIMS....
        # could try the pH 7 version of this?
        # TOP OF LEFT HEMISPHERE IS CLIPPED OFF A BIT
        # '/media/user/20TB_HDD_NAS_2/20240416_M311_REDO_MoE_CUPRIZONE_6wks_RECOVERY_3wks_SHIELD_CUBIC_7d_RIMS_RI_1496_3d_100perc_488_80msec/'


        # P60 (current) - no autofluor sub
        # '/media/user/20TB_HDD_NAS/20231117_M115_MoE_P56_5days_delip_RIMS_RI_1487_7days_5x_80perc_laser_REDO_WITH_SUNFLOWER/',







        # 96rik analysis
        # '/media/user/20TB_HDD_NAS/20240209_M248_96rik_SHIELD_CUBIC_7d_RIMS_3d_RI_1493_sunflower/',

        # less bloody brain?
        # '/media/user/8TB_HDD/20240216_M246_96rik_with_nanobodies_SHIELD_CUBIC_RIMS_RI_1493_sunflow/',


        # ANYA BRAIN
        # '/media/user/20TB_HDD_NAS_2/20240813_AK11_96rik_p30_whole_brain_rims_80power/11_testing_f_p34/fused/', #Done
        # '/media/user/20TB_HDD_Anya/20240625_Brain_13_f_p114_Lncol1_5x_488_80p_638_20p_SHIELD_CUBIC_RIMS/fused/', #Done
        # '/media/user/20TB_HDD_Anya/20240625_Brain_16_f_p443_Lncol1_5x_488_80p_638_20p_SHIELD_CUBIC_RIMS/fused/', #Done
        # '/media/user/20TB_HDD_Anya/20240625_Brain_14_f_p114_Lncol1_5x_488_80p_638_20p_SHIELD_CUBIC_RIMS/fused/' #Done
        # '/media/user/20TB_HDD_Anya/20240625_Brain_15_m_p515_Lncol1_5x_488_80p_638_20p_SHIELD_CUBIC_RIMS/fused/' #Done
        # /media/user/20TB_HDD_Anya/20240625_Brain_13_f_p114_Lncol1_5x_488_80p_638_20p_SHIELD_CUBIC_RIMS/fused/' #Done
        # '/media/user/20TB_HDD_NAS/20240209_M248_96rik_SHIELD_CUBIC_7d_RIMS_3d_RI_1493_sunflower/' #Done
        # '/media/user/20TB_HDD_Anya/20250427_L001_17_ak_ra_f9_32_80p_488_20p_638_30ms/fused/', #Done
        # '/media/user/20TB_HDD_Anya/20250502_L001_19_ak_ra_mlr_p157_80p_488_20p_638_30ms/fused/', #Done
        # '/media/user/20TB_HDD_Anya/20250427_L001_20_ak_ra_f3_57_80p_488_20p_638_30ms/fused/', #Done
        # '/media/user/20TB_HDD_Anya/20250502_L001_21_ak_ra_m6_p57_bad_perf/fused', #Done
        # '/media/user/20TB_HDD_Anya/20250502_L001_18_ak_ra_mll_157_85p_488_20p_638_30ms_reimage_after_ob_error/fused', #Done
        # '/media/user/20TB_HDD_Anya/20250502_L001_22_ak_ra_f2_p488/fused/', #Done
        # '/media/user/20TB_HDD_Anya/20250502_L001_23_ak_ra_m13_p32_488_80p_638_20p/fused/', #Done
        # '/media/user/20TB_HDD_Anya/20250502_L001_24_ak_ra_f3_488_85p_488_20p_638_30ms/fused/',
        # Had to restart 24 after Tiger reset the computer




        # FOR 5xFAD
        # '/media/user/20TB_HDD_Anya/20241205_M217_MoE_5xFAD_CongoRed_SHIELD_CUBIC_RIMS_RI_14917_1day/',



        # '/media/user/20TB_HDD_Anya/20241213_M243_MoE_5xFAD_CongoRed1-100_SHIELD_CUBIC_RIMS_RI_14911_2days_40perc_laser/',  ### FOV 311 for congored





        # '/media/user/20TB_HDD_Anya/20241214_M244_MoE_5xFAD_CongoRed1-100_SHIELD_CUBIC_RIMS_RI_14911_3days_40perc_laser/',

        # '/media/user/20TB_HDD_Anya/20241218_M234_MoE_5xFAD_CongoRed1-100_SHIELD_CUBIC_RIMS_RI_14911_2days_40perc_laser/',

        # Final 5xFAD to be run
        # '/media/user/20TB_HDD_Anya/20241219_M235_MoE_5xFAD_CongoRed1-100_SHIELD_CUBIC_RIMS_RI_14911_2days_40perc_laser/',



        # '/media/user/20TB_HDD_Anya/20241215_M296_MoE_6mos_CongoRed1-100_SHIELD_CUBIC_RIMS_RI_14911_4days_40perc_laser/',


        # '/media/user/20TB_HDD_Anya/20241218_M297_MoE_6mos_CongoRed1-100_SHIELD_CUBIC_RIMS_RI_14911_6days_40perc_laser/',

        # '/media/user/20TB_HDD_Anya/20241214_M304_MoE_6mos_CongoRed1-100_SHIELD_CUBIC_RIMS_RI_14911_3days_40perc_laser/',

        # '/media/user/20TB_HDD_Anya/20241214_M242_MoE_control_6mos_CongoRed1-100_SHIELD_CUBIC_RIMS_RI_14911_3days_40perc_laser/',



        ##################################################################################################################

        # For new .n5 files!
        # '/media/user/4TB_HDD/20231012_M230_MoE_PVCre_SHIELD_delip_RIMS_RI_1500_3days_5x/'
        # '/media/user/8TB_HDD/20231031_M229_MoE_PVCre_P56_SHIELD_delp_RIMS_50perc_then_100perc_expanded_slightly_more_5x/',

        # '/media/user/8TB_HDD/20231115_M124_MoE_CasprtdT_Cuprizone_6wk__SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER/',

        # '/media/user/8TB_HDD/20231115_M139_MoE_CasprtdT_Cuprizone_6wk__SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER/',

        # '/media/user/8TB_HDD/20231117_M222_96rik_Evans_Blue_SHIELD_RIMS_RI_1499_5x_80perc_laser_SUNFLOWER/',




        # '/media/user/c0781205-1cf9-4ece-b3d5-96dd0fbf4a78/20231218_M216_MoE_control_for_5xFAD_RIMS_RI_1489_sunflower_80perc/',


        # '/media/user/Tx_LS_Data_5/20231116_M169_MoE_CD1_SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER_WARPED/M169_CD1_fused/',

        # M154 SUPER BLOODY
        # '/media/user/8TB_HDD/20231222_M154_FVB_delip_reused_RIMS_weird_no_sunflower1st_time_RIMS_RI_1489_sunflower_80perc/',

        # '/media/user/Tx_LS_Data_5/20231031_M229_MoE_PVCre_P56_SHIELD_delp_RIMS_50perc_then_100perc_expanded_slightly_more_5x_NOCZI/',

        # '/media/user/8TB_HDD/20240109_M27_MoE_OLD_SHIELD_CUBIC_RI_RIMS_14926_sunflow/',

        # For 12 months - brain is a bit warped though
        # '/media/user/20TB_HDD_NAS/20240229_M305_MoE_12mos_SHIELD_CUBIC_6d_RIMS_2d_RI_new_14925_sunflow/',

        # For 12 months - with Tie2Cre;Ai9
        # '/media/user/20TB_HDD_NAS/20240229_M274_MoE_Tie2Cre_Ai9_12mos_SHIELD_CUBIC_6d_RIMS_1d_RI_new_unsure_sunflow/',


        # ### For 12 months
        # '/media/user/20TB_HDD_NAS_2/20240420_M306_REDO_PB_washed_1_year_SHIELD_CUBIC_RI_RIMS_1496_1d_after_wash_100perc_488_50perc_638_100msec/',


        # %% REDO DIM DATA so hopefully not dim anymore


        # 6wks cup + 3wks recovery
        # '/media/user/20TB_HDD_NAS_2/20240416_M311_REDO_MoE_CUPRIZONE_6wks_RECOVERY_3wks_SHIELD_CUBIC_7d_RIMS_RI_1496_3d_100perc_488_80msec/',





        # %% Jaime data
        '/media/user/Elements/B711_Jaime_MoE_PVCre_Ai9/'

        # '/media/user/Elements/A412/fused/',

        # '/media/user/Elements/A457/fused/'

        # '/media/user/Elements/C407/fused/'

        # '/media/user/Elements/C459/fused/'

        # '/media/user/Elements/C460/fused/'

        # '/media/user/Elements/C461/fused/'

        # '/media/user/Elements/C464/fused/'

    ]

    # ALWAYS LEAVE THIS ON --- SLOWER BUT MAKES SURE YOU GET EVERY CELL
    lncol1 = True
    print('IS THIS 96rik data??? OR CUPRIZONE??? - double check to change thresh')

    # ANYA --- for vessel sub turn to 1
    autofluor_sub = 0  # TO SUBTRACT OUT AUTOFLUORESCENCE

    congoRed = 0

    # If Jaime, turn to 1, if else, turn to 0
    jaime = 0
    # use 0tx8 and FOV 311

    # if autofluor_sub:
    #     print('Training_APOC_classifier')
    #     pixel_classifier = train_APOC_blood()

    # Initiate poolThread
    poolThread_load_data = ThreadPool(processes=1)
    # poolThread_post_process = ThreadPool(processes=2)

    """ Loop through all the folders and do the analysis!!!"""
    for input_path in list_folder:

        """ For testing ILASTIK images """
        images = glob.glob(os.path.join(
            input_path, '*.n5'))    # can switch this to "*truth.tif" if there is no name for "input"
        images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
        examples = [dict(input=i, truth=i.replace('.n5', '.xml'), ilastik=i.replace(
            '.tif', '_single_Object Predictions_.tiff')) for i in images]

        input_name = images[0]
        filename = input_name.split('/')[-1].split('.')[0:-1]
        filename = '.'.join(filename)

        sav_dir = input_path + '/' + filename + '_MaskRCNN_patches'

        try:
            # Create target Directory
            os.mkdir(sav_dir)
            print("\nSave directory ", sav_dir,  " Created ")
        except FileExistsError:
            print("\nSave directory ", sav_dir,  " already exists")

        sav_dir = sav_dir + '/'

        # Required to initialize all
        for file_num in range(len(examples)):

            """ TRY INFERENCE WITH PATCH-BASED analysis from TORCHIO """
            with torch.set_grad_enabled(False):  # saves GPU RAM
                input_name = examples[file_num]['input']
                # input_im = tiff.imread(input_name)

                # import z5py
                # with z5py.File(input_name, "r") as f:

                f = z5py.File(examples[file_num]['input'], "r")

                if jaime:
                    dset = f['setup2/timepoint0/s0']
                else:
                    dset = f['setup0/timepoint0/s0']

                # Get red data for autofluorescence subtraction as well
                if autofluor_sub:
                    dset_red = f['setup1/timepoint0/s0']

                # For congo red analysis
                if congoRed:
                    dset_congo = f['setup2/timepoint0/s0']

                # lowest_res = f['t00000']['s00']['7']['cells']
                # highest_res = f['t00000']['s00']['0']['cells']

                # dset = highest_res

                # channel 2
                # highest_res = f['t00000']['s01']['0']['cells']
                coords_df = pd.DataFrame(columns=[
                                         'offset', 'block_num', 'Z', 'X', 'Y', 'Z_scaled', 'X_scaled', 'Y_scaled', 'equiv_diam', 'vol'])

                """ Or save to memmapped TIFF first... """
                print('creating memmap save file on disk')
                # memmap_save = tiff.memmap(sav_dir + 'temp_SegCNN.tif', shape=dset.shape, dtype='uint8')
                # memmap_save = tiff.memmap('/media/user/storage/Temp_lightsheet/temp_SegCNN_' + str(file_num) + '_.tif', shape=dset.shape, dtype='uint8')

                """ Figure out how many blocks need to be generated first, then loop through all the blocks for analysis """
                print('Extracting block sizes')

                """ Or use chunkflow instead??? https://pychunkflow.readthedocs.io/en/latest/tutorial.html"""
                im_size = np.shape(dset)
                depth_imL = im_size[0]
                heightL = im_size[1]
                widthL = im_size[2]

                total_blocks = 0
                all_xyz = []

                """ These should be whole values relative to the chunk size so that it can be uploaded back later! """
                # quad_size = 128 * 10
                # quad_depth = 64 * 4

                Lpatch_size = 128 * 10
                Lpatch_depth = 64 * 4

                # quad_size = round(input_size * 1/XY_scale * 3)
                # quad_depth = round(depth * 1/XY_scale * 3)

                # overlap_percent = 0
                # Add padding to input_im

                # print('Total num of patches: ' + str(factorx * factory * factorz))

                all_xyzL = []

                thread_post = 0
                called = 0

                for z in range(0, depth_imL, round(Lpatch_depth)):
                    # if z + Lpatch_depth > depth_imL:  continue

                    for x in range(0, widthL, round(Lpatch_size)):
                        # if x + Lpatch_size > widthL:  continue

                        for y in range(0, heightL, round(Lpatch_size)):
                            # if y + Lpatch_size > heightL: continue

                            # print([x, y, z]
                            all_xyzL.append([x, y, z])

                # how many total blocks to analyze:
                # print(len(all_xyzL))

                def get_im(dset, s_c, Lpatch_depth, Lpatch_size):

                    # tic = time.perf_counter()

                    # If nearing borders of image, prevent going out of bounds!
                    z_top = s_c[2] + Lpatch_depth
                    if z_top >= dset.shape[0]:
                        z_top = dset.shape[0]

                    y_top = s_c[1] + Lpatch_size
                    if y_top >= dset.shape[1]:
                        y_top = dset.shape[1]

                    x_top = s_c[0] + Lpatch_size
                    if x_top >= dset.shape[2]:
                        x_top = dset.shape[2]

                    input_im = dset[s_c[2]:z_top, s_c[1]:y_top, s_c[0]:x_top]
                    og_shape = input_im.shape

                    # toc = time.perf_counter()
                    print('loaded asynchronously')

                    # print(f"Opened subblock in {toc - tic:0.4f} seconds")

                    return input_im, og_shape

                """ Then loop through """
                # for id_c, s_c in enumerate(all_xyzL):

                # for continuing the run

                # FOR M91 first run:
                # SKIPPED OVER 135
                # also skipped 142 - 160
                # also 220, 248, 262, 277, 293, 352, 381, 391, 453,468, 476
                # 500

                # 500, 513, 514

                # M127 had problem with nan bounding boxes on id_c = 192   - 2024 June 6

                # ANYA - make sure not wrong - can input tile number to restart- Change it both here and in the if statement below
                for id_c in range(0, len(all_xyzL)):  # 192, 304,336, 243
                    # for id_c in range(299, 320):

                    # id_c = 100

                    s_c = all_xyzL[id_c]

                    # for debug:
                    # s_c = all_xyz[10]
                    tic = time.perf_counter()

                    # input_im, og_shape = get_im(dset, s_c, Lpatch_depth, Lpatch_size)
                    # plot_max(input_im)

                    # Load first tile normally, and then the rest as asynchronous processes

                    # ANYA - input tile number to restart - Change this and the above for statement
                    # 313 for otx18 #336 for M428 (disk space issue)
                    
###Inclusion of N5 loading error  - Jaime 
                    try: 
                                        
                        if id_c == 0:
                            input_im, og_shape = get_im(
                                dset, s_c, Lpatch_depth, Lpatch_size)
    
                            # input_im = input_im * 1.8
                            # input_im = np.asarray(input_im, dtype=np.uint16)
    
                            print('loaded normally')
    
                        else:  # get tile from asynchronous processing instead!
                            # zzz
    
                            # if id_c in [135, 142, 160, 220, 248, 262, 277, 293, 352, 381, 391, 453, 468, 476, 500, 513]:
                            #         print('skip for M91')
                            #         continue
                            # else:
                            # zzz
                            # get the return value from your function.
                            input_im, og_shap = async_result.get()
    
                            # poolThread_load_data.close()
                            # poolThread_load_data.join()
    
                        # get NEXT tile asynchronously!!!
                        if id_c + 1 < len(all_xyzL):  # but stop once it's reached the end
    
                            # if id_c + 1 in [135, 142, 160, 220, 248, 262, 277, 293, 352, 381, 391, 453, 468, 476, 500, 513]:
                            #     print('skip for M91')
    
                            # else:
                            async_result = poolThread_load_data.apply_async(
                                get_im, (dset, all_xyzL[id_c + 1], Lpatch_depth, Lpatch_size))
    
                        # if id_c in [135, 142, 160, 220, 248, 262, 277, 293, 352, 381, 391, 453, 468, 476, 500, 513]:
                        #         print('skip for M91')
                        #         continue
    
                        print('one loop')
                    
                    except RuntimeError as e:
                        print(f"[{id_c}] Skipping block due to decompression error: {e}")
                        
                        if id_c + 1 < len(all_xyzL):
                            async_result = poolThread_load_data.apply_async(get_im, (dset, all_xyzL[id_c + 1], Lpatch_depth, Lpatch_size))
                 
                        continue
                    
                    except Exception as e:
                        print(f"[{id_c}] Skipping block due to unexpected error: {e}")
                        continue
### End

                    toc = time.perf_counter()
                    print(f"\nOpened subblock in {toc - tic:0.4f} seconds")

                    """ Detect if blank in uint16 """
                    if lncol1:
                        num_voxels = len(np.where(input_im > 120)[0])
                    else:
                        num_voxels = len(np.where(input_im > 300)[0])
                    if num_voxels < 10000:
                        print('skipping: ' + str(s_c))
                        print('num voxels with signal: ' + str(num_voxels))

                        # time.sleep(10)
                        continue

                    if congoRed:
                        congo, og_shape = get_im(
                            dset_congo, s_c, Lpatch_depth, Lpatch_size)

                        input_im = np.asarray(input_im, dtype=np.uint16)
                        tiff.imwrite(sav_dir + filename + '_' +
                                     str(id_c) + '_green_im.tif', input_im)

                        congo = np.asarray(congo, dtype=np.uint16)
                        tiff.imwrite(sav_dir + filename + '_' +
                                     str(id_c) + '_CONGO_im.tif', congo)

                        red, og_shape = get_im(
                            dset_red, s_c, Lpatch_depth, Lpatch_size)
                        red = np.asarray(red, dtype=np.uint16)
                        tiff.imwrite(sav_dir + filename + '_' +
                                     str(id_c) + '_red_im.tif', red)

                    """ ### Run in a way that only re-does the missed ones """
                    if os.path.isfile(sav_dir + filename + '_' + str(int(id_c)) + '_df.pkl'):
                        continue

                    # Get red data for autofluor subtraction
                    if autofluor_sub:
                        red, og_shape = get_im(
                            dset_red, s_c, Lpatch_depth, Lpatch_size)

                        input_im = np.asarray(input_im, dtype=np.uint16)
                        tiff.imwrite(sav_dir + filename + '_' +
                                     str(id_c) + '_green_im.tif', input_im)

                        red = np.asarray(red, dtype=np.uint16)
                        tiff.imwrite(sav_dir + filename + '_' +
                                     str(id_c) + '_red_im.tif', red)

                        div = red/input_im

                        # set an upper bound?...
                        bound = 0.2
                        c = np.transpose(np.where(div < bound))
                        div[c[:, 0], c[:, 1], c[:, 2]] = bound

                        div2 = input_im/div

                        div2[np.isnan(div2)] = 0  # get rid of nans
                        div2[np.isinf(div2)] = 0  # get rid of nans

                        input_im = div2

                        # # """ Updated all code to ITK-elastix which is so much easier to install and use"""
                        # # def register_ELASTIX_ITK(fixed_im, moving_im, reg_type='affine'):

                        # #     import itk as itk
                        # #     reg_type='affine'

                        # #     fixed_image = np.asarray(fixed_im, dtype=np.float32)
                        # #     moving_image = np.asarray(moving_im, dtype=np.float32)

                        # #     parameter_object = itk.ParameterObject.New()
                        # #     parameter_map_rigid = parameter_object.GetDefaultParameterMap('translation')
                        # #     parameter_object.AddParameterMap(parameter_map_rigid)

                        # #     if reg_type == 'affine':
                        # #         parameter_map_rigid = parameter_object.GetDefaultParameterMap('affine')
                        # #         parameter_object.AddParameterMap(parameter_map_rigid)

                        # #     else:
                        # #         print('required reg_type: "affine" or None')

                        # #     ### Choose interpolator order
                        # #     parameter_map_rigid['FinalBSplineInterpolationOrder'] = ['0']
                        # #     parameter_object.AddParameterMap(parameter_map_rigid)

                        # #     # Call registration function
                        # #     registered_im, result_transform_parameters = itk.elastix_registration_method(
                        # #         fixed_image, moving_image,
                        # #         parameter_object=parameter_object)

                        # #     registered_im = np.asarray(registered_im, dtype=np.uint16)
                        # #     return registered_im

                        # # print('REGISTERING AUTOFLUORESCENCE CHANNEL')
                        # # registered_im = register_ELASTIX_ITK(input_im, red, reg_type='affine')

                        # # zzz

                        # # mask = np.copy(registered_im)
                        # # mask[mask < 200] = 0

                        # #sub_im = np.asarray(input_im, dtype=np.int16) - (np.abs(np.asarray(registered_im, dtype=np.int16)))

                        # # sub_im[sub_im < 0] = 0

                        # # ### A LITTLE HACK - since affine reg above causes shifts in volume, want to zero out all the edges that have nothing
                        # # ### fix this in the future by sending a larger FOV overall!!!
                        # # sub_im[registered_im == 0] = 0

                        # # sub_im = np.asarray(sub_im, dtype=np.uint16)

                        # ### SUBTRACT MEDIAN SO GREEN AND RED ARE NORMALIZED TO EACH OTHER??? or use background???

                        # seg_no_clean = APOC_by_crop(pixel_classifier, ch1_im=input_im, ch2_im=red, quad_depth=256, quad_size=640)

                        # ### Mask out autofluor channel and then subtract it out from green image
                        # masked_ch2 = np.copy(red)
                        # masked_ch2[seg_no_clean == 0] = 0

                        # sub_im = np.copy(input_im)
                        # sub_im = np.asarray(sub_im, dtype=np.int16) - masked_ch2
                        # sub_im[sub_im < 0] = 150

                        # tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_green.tif', np.asarray(input_im, dtype=np.uint16))
                        # tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_red.tif', np.asarray(red, dtype=np.uint16))

                        # tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_seg_blood.tif', np.asarray(seg_no_clean, dtype=np.uint16))
                        # tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_sub_im.tif', np.asarray(sub_im, dtype=np.uint16))

                        # plot_max(red)
                        # plot_max(seg_no_clean)

                        # ### set as new input_im
                        # input_im = sub_im

                        # # save_im = np.expand_dims(input_im, axis=0)
                        # # save_im = np.expand_dims(save_im, axis=2)
                        # # tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +'_red.tif', np.asarray(save_im, dtype=np.uint16),
                        # #                        imagej=True, #resolution=(1/XY_res, 1/XY_res),
                        # #                        metadata={'spacing':1, 'unit': 'um', 'axes': 'TZCYX'})

                    print('Analyzing: ' + str(s_c))
                    print('Which is: ' + str(id_c) +
                          ' of total: ' + str(len(all_xyzL)))

                    """ Start inference on volume """
                    tic = time.perf_counter()

                    import warnings
                    overlap_percent = 0.1

                    # %% MaskRCNN analysis!!!
                    """ Analyze each block with offset in all directions """
                    # print('Starting inference on volume: ' + str(file_num) + ' of total: ' + str(len(examples)))

                    # Define patch sizes
                    patch_size = 128
                    patch_depth = 16

                    # Define overlap and focal cube to remove cells that fall within this edge
                    overlap_pxy = 14
                    overlap_pz = 3
                    step_xy = patch_size - overlap_pxy * 2
                    step_z = patch_depth - overlap_pz * 2

                    focal_cube = np.ones([patch_depth, patch_size, patch_size])
                    focal_cube[overlap_pz:-overlap_pz, overlap_pxy:-
                               overlap_pxy, overlap_pxy:-overlap_pxy] = 0
                    focal_cube = np.moveaxis(focal_cube, 0, -1)

                    thresh = 0.9

                    # thresh = 0.9
                    # cf.merge_3D_iou = thresh

                    im_size = np.shape(input_im)
                    width = im_size[1]
                    height = im_size[2]
                    depth_im = im_size[0]

                    # Add padding to input_im
                    factorx = 0
                    while factorx < width/step_xy:
                        factorx += 1
                    end_padx = (factorx * step_xy) - width

                    factory = 0
                    while factory < height/step_xy:
                        factory += 1
                    end_pady = (factory * step_xy) - height

                    factorz = 0
                    while factorz < depth_im/step_z:
                        factorz += 1
                    end_padz = (factorz * step_z) - depth_im

                    print('Total num of patches: ' +
                          str(factorx * factory * factorz))
                    new_dim_im = np.zeros([overlap_pz*2 + depth_im + end_padz, overlap_pxy *
                                          2 + width + end_padx, overlap_pxy*2 + height + end_pady])
                    new_dim_im[overlap_pz: overlap_pz + depth_im, overlap_pxy: overlap_pxy +
                               width, overlap_pxy: overlap_pxy + height] = input_im
                    input_im = new_dim_im

                    im_size = np.shape(input_im)
                    width = im_size[1]
                    height = im_size[2]
                    depth_im = im_size[0]

                    # Define empty items
                    box_coords_all = []
                    total_blocks = 0
                    segmentation = np.zeros([depth_im, width, height])
                    # colored_im = np.zeros([depth_im, width, height])

                    # split_seg = np.zeros([depth_im, width, height])
                    all_xyz = []
                    all_blknum = []

                    all_patches = []
                    all_output = []

                    batch_size = 1
                    batch_im = []
                    batch_xyz = []

                    debug = 0

                    # SET THE STEP SIZE TO BE HALF OF THE IMAGE SIZE
                    # step_z = patch_depth/2
                    # step_xy = patch_size/2

                    # add progress bar
                    # estimate number of blocks
                    pbar = tqdm(total=factorx * factory * factorz,
                                desc="Loading",
                                ncols=75)

                    # %% START MASK-RCNN analysis
                    for z in range(0, depth_im + patch_depth, round(step_z)):
                        if z + patch_depth > depth_im:
                            continue

                        for x in range(0, width + patch_size, round(step_xy)):
                            if x + patch_size > width:
                                continue

                            for y in range(0, height + patch_size, round(step_xy)):
                                if y + patch_size > height:
                                    continue

                                quad_intensity = input_im[z:z + patch_depth,
                                                          x:x + patch_size, y:y + patch_size]
                                quad_intensity = np.moveaxis(
                                    quad_intensity, 0, -1)
                                quad_intensity = np.asarray(np.expand_dims(np.expand_dims(
                                    quad_intensity, axis=0), axis=0), dtype=np.float16)

                                # FOR SOME REASON GETTING "INF" --- maybe detector maxed out values? - set to max instead
                                if len(np.where(quad_intensity == np.inf)[0]) > 0:
                                    print('INF DETECTED')
                                quad_intensity[quad_intensity == np.inf] = np.max(
                                    quad_intensity[quad_intensity != np.inf])

                                """ Detect if blank in uint16 """
                                if lncol1:  # lower thresh for 96rik
                                    num_voxels = len(
                                        np.where(quad_intensity > 120)[0])

                                    if autofluor_sub:
                                        num_voxels = len(
                                            np.where(quad_intensity > 20)[0])
                                    if num_voxels < 50:
                                        # print('skipping: ' + str(s_c))
                                        # print('num voxels with signal: ' + str(num_voxels))
                                        # update pbar even if skipped!
                                        pbar.update(1)
                                        continue

                                else:
                                    num_voxels = len(
                                        np.where(quad_intensity > 300)[0])

                                    if autofluor_sub:
                                        num_voxels = len(
                                            np.where(quad_intensity > 150)[0])
                                    if num_voxels < 300:
                                        # print('skipping: ' + str(s_c))
                                        # print('num voxels with signal: ' + str(num_voxels))
                                        # update pbar even if skipped!
                                        pbar.update(1)
                                        continue

                                # %% NORMALIZATION
                                # First normalize to be between 0 - 1

                                # quad_intensity = quad_intensity/65535

                                norm_data = (quad_intensity -
                                             mean_data)/std_data

                                quad_intensity = norm_data

                                if len(batch_im) > 0:
                                    batch_im = np.concatenate(
                                        (batch_im, quad_intensity))
                                else:
                                    batch_im = quad_intensity

                                batch_xyz.append([x, y, z])

                                # print(total_blocks)
                                total_blocks += 1

                                if total_blocks % batch_size == 0:

                                    batch = {'data': batch_im, 'seg': np.zeros([batch_size, 1, patch_size, patch_size, patch_depth]),
                                             'class_targets': np.asarray([]), 'bb_targets': np.asarray([]),
                                             'roi_masks': np.zeros([batch_size, 1, 1, patch_size, patch_size, patch_depth]),
                                             'patient_bb_target': np.asarray([]), 'original_img_shape': quad_intensity.shape,
                                             'patient_class_targets': np.asarray([]), 'pid': ['0']}

                                    # seg preds are only seg_logits! need to take argmax.
                                    output = net.test_forward(
                                        batch, main_brain=True)

                                    all_output.append(output)

                                    # Reset batch
                                    batch_im = []
                                    batch_xyz = []

                                    # patch = np.moveaxis(quad_intensity, -1, 1)
                                    # save = np.concatenate((patch, seg_im), axis=2)

                                    all_xyz.append([x, y, z])
                                    all_blknum.append(total_blocks)

                                    pbar.update(1)

                    pbar.close()
                    # zzz

                    toc = time.perf_counter()

                    print(f"MaskRCNN analysis in {toc - tic:0.4f} seconds")

                    # %% Process in parallel outputs to extract result_dict

                    def parse_outputs_to_coords(kwargs):
                        # print(kwargs)

                        output = kwargs[0]
                        xyz_patch = kwargs[1]
                        thresh = kwargs[2]
                        blk_num = kwargs[3]

                        # print(output.shape)
                        # print(blk_num)

                        results_dict = net.get_results(img_shape=output[0], detections=output[1],
                                                       detection_masks=output[2], return_masks=output[3])
                        im_shape = output[0]

                        # Add box patch factor
                        new_box_list = []
                        tmp_check = np.zeros(im_shape[2:])

                        for bid, box in enumerate(results_dict['boxes'][0]):
                            """ 

                                Some bounding boxes have no associated segmentations!!! Skip these

                            """
                            if len(box['mask_coords']) == 0:
                                continue

                            # Only add if above threshold
                            if box['box_score'] > thresh:
                                c = box['box_coords']

                                box_centers = [
                                    (c[ii] + c[ii + 2]) / 2 for ii in range(2)]
                                box_centers.append((c[4] + c[5]) / 2)

                                # factor = np.mean([norm.pdf(bc, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8 for bc, pc in \
                                #                   zip(box_centers, np.array(quad_intensity[0][0].shape) / 2)]
                                # slightly faster call
                                pc = np.array(im_shape[2:]) / 2
                                factor = np.mean(
                                    [norm.pdf(box_centers, loc=pc, scale=pc * 0.8) * np.sqrt(2 * np.pi) * pc * 0.8])

                                box['box_patch_center_factor'] = factor
                                new_box_list.append(box)

                        results_dict['boxes'] = [new_box_list]
                        # results_dict = results_dict

                        if 'seg_preds' in results_dict.keys():
                            results_dict['seg_preds'] = np.argmax(
                                results_dict['seg_preds'], axis=1)[:, np.newaxis]
                        # results_dict['colored_boxes'] = np.expand_dims(results_dict['colored_boxes'][:, 1, :, :, :], axis=0)

                        # Add to segmentation
                        seg_im = np.moveaxis(results_dict['seg_preds'], -1, 1)

                        # get coords so can plot into segmentation later
                        coords = np.transpose(
                            np.where(seg_im[0, :, 0, :, :] > 0))

                        coords[:, 0] = coords[:, 0] + xyz_patch[2]
                        coords[:, 1] = coords[:, 1] + xyz_patch[0]
                        coords[:, 2] = coords[:, 2] + xyz_patch[1]

                        # color_im = np.moveaxis(results_dict['colored_boxes'], -1, 1)
                        # segmentation[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] = segmentation[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] + seg_im[0, :, 0, :, :]
                        # colored_im[z:z + patch_depth,  x:x + patch_size, y:y + patch_size] = color_im[0, :, 0, :, :]

                        # add option for larger batching?
                        # for bs in range(batch_size):

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

                        # patch_im = batch_im[bs][0]
                        # patch_im = np.moveaxis(patch_im, -1, 0)   ### must be depth first

                        # save memory by deleting seg_preds
                        results_dict['seg_preds'] = []

                        # all_patches.append({ 'results_dict': results_dict, 'total_blocks': bs % total_blocks + (total_blocks - batch_size),
                        #                       #'focal_cube':focal_cube, 'patch_im': patch_im, 'box_vert':box_vert, 'mask_coords':mask_coords
                        #                       'xyz':xyz_patch})

                        # all_patches.append({ 'results_dict': results_dict, 'total_blocks': total_blocks,
                        #                      #'focal_cube':focal_cube, 'patch_im': patch_im, 'box_vert':box_vert, 'mask_coords':mask_coords
                        #                      'xyz':xyz_patch})

                        return results_dict, blk_num, xyz_patch, coords

                    tic = time.perf_counter()
                    # add x,y,z and save segmentation
                    kwargs = zip(all_output, all_xyz, [
                                 thresh] * len(all_xyz), all_blknum)

                    # kwargs = {'output':all_output, 'xyz_patch':all_xyz, 'thresh':[thresh]*len(all_xyz), 'total_blocks':[total_blocks]*len(all_xyz)}

                    exec_results = concurrent.futures.ThreadPoolExecutor(10)
                    results = list(tqdm(exec_results.map(
                        parse_outputs_to_coords, kwargs), total=len(all_output)))
                    exec_results.shutdown(wait=True)

                    toc = time.perf_counter()
                    print(f"Parse output coords in {toc - tic:0.4f} seconds")

                    seg_coords = []
                    for result in results:
                        all_patches.append({'results_dict': result[0], 'total_blocks': result[1],
                                            # 'focal_cube':focal_cube, 'patch_im': patch_im, 'box_vert':box_vert, 'mask_coords':mask_coords
                                            'xyz': result[2]})
                        seg_coords.append(result[3])

                    # plot to make segmentation array with overlapping regions showing
                    for coords in seg_coords:
                        # create segmentation image
                        segmentation[coords[:, 0], coords[:, 1], coords[:, 2]
                                     ] = segmentation[coords[:, 0], coords[:, 1], coords[:, 2]] + 1

                    # pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                    # result = pool.map(parse_outputs_to_coords, cc)
                    # pool.shutdown(wait=True)

                    # def parse_outputs_to_coords(output, xyz_patch, thresh, total_blocks):

                    # zzz

                    # %% Post-process boxes

                    # First save the files and remove them from RAM so we don't max out
                    print('saving files')
                    im_size = np.shape(input_im)

                    filename = input_name.split('/')[-1].split('.')[0:-1]
                    filename = '.'.join(filename)

                    input_im = np.expand_dims(input_im, axis=0)
                    input_im = np.expand_dims(input_im, axis=2)
                    tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) + '_input_im.tif', np.asarray(input_im, dtype=np.uint16),
                                 # resolution=(1/XY_res, 1/XY_res),
                                 imagej=True,
                                 metadata={'spacing': 1, 'unit': 'um', 'axes': 'TZCYX'})

                    segmentation = np.asarray(segmentation, np.uint8)
                    tiff.imwrite(sav_dir + filename + '_' + str(int(id_c)) +
                                 '_segmentation_overlap3.tif', segmentation)
                    # post_process_async(cf, input_im, segmentation, input_name, sav_dir, all_patches, patch_size, patch_depth, id_c, focal_cube, s_c, debug)

                    # zzz
                    # Then call asynchronous post-processing to sort out boxes
                    if called:
                        executor.submit(post_process_async, cf, all_patches, im_size, filename, sav_dir, overlap_pxy, overlap_pz,
                                        patch_size, patch_depth, id_c, focal_cube, s_c=s_c, debug=debug)

                    else:
                        executor = concurrent.futures.ThreadPoolExecutor(
                            max_workers=4)

                        executor.submit(post_process_async, cf, all_patches, im_size, filename, sav_dir, overlap_pxy, overlap_pz,
                                        patch_size, patch_depth, id_c, focal_cube, s_c=s_c, debug=debug)

                        called = 1

                    # zzz
                    # clean-up
                    segmentation = []
                    input_im = []
                    all_patches = []

                    # %% Output to memmap array

                    """ Output to memmmapped arr """

                    # tic = time.perf_counter()
                    # memmap_save[s_c[2]:s_c[2] + Lpatch_depth, s_c[1]:s_c[1] + Lpatch_size, s_c[0]:s_c[0] + Lpatch_size] = segmentation
                    # memmap_save.flush()

                    # toc = time.perf_counter()

                    # print(f"Save in {toc - tic:0.4f} seconds")

                    # tic = time.perf_counter()

    print('\n\nSegmented outputs saved in folder: ' + sav_dir)

    # REMEMBER TO CLOSE POOL THREADS
    poolThread_load_data.close()
    # poolThread_post_process.close()
